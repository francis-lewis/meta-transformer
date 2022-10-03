import copy
import os
import random
import sys
import numpy as np

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torchvision

EXPERIMENT_ROOT = os.path.expanduser("~/Developer/experiments")

def proc_setup(rank, num_proc):
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '12345'

  dist.init_process_group("nccl", rank=rank, world_size=num_proc)

def cleanup():
  dist.destroy_process_group()

def set_random_seeds(random_seed=0):

  torch.manual_seed(random_seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  np.random.seed(random_seed)
  random.seed(random_seed)

def define_finetune_model(base_model, num_classes=10, head_layer_name='fc', finetune_base=False):
  if not finetune_base:
    for param in base_model.parameters():
      param.requires_grad = False

  num_in_feats = getattr(base_model, head_layer_name).in_features
  setattr(base_model, head_layer_name, torch.nn.Linear(num_in_feats, num_classes))

  return base_model

def get_data_loaders(dataset, transform, batch_size=64, root=os.path.join(EXPERIMENT_ROOT, 'data'), num_workers=0, is_distributed=False):

  train_set = dataset(root=root, train=True, download=False, transform=transform)
  test_set = dataset(root=root, train=False, download=False, transform=transform)

  train_sampler = None
  if is_distributed:
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset=train_set)

  train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
  test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

  return train_loader, test_loader

def test(model, device, test_loader):

  model.to(device)
  model.eval()

  correct = 0
  total = 0
  # since we're not training, we don't need to calculate the gradients for our outputs
  with torch.no_grad():
      for data in test_loader:
          images, labels = data[0].to(device), data[1].to(device)
          outputs = model(images)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()

  accuracy = correct / total
  return accuracy

def train(model, loss_fn, optimizer, device, train_loader, epoch=1, report_every_n=2000, min_num_reports=5):

  model.to(device)
  model.train()

  total_batches = len(train_loader)
  if total_batches / report_every_n < min_num_reports:
    report_every_n = total_batches // min_num_reports

  running_loss = 0.0
  final_loss = None
  for batch_num, data in enumerate(train_loader, 1):
    inputs, labels = data[0].to(device), data[1].to(device)

    optimizer.zero_grad()

    outputs = model(inputs)
    loss = loss_fn(outputs, labels)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()

    if batch_num % report_every_n == 0:
      print(f'[{epoch}, {batch_num:5d}] loss: {running_loss / report_every_n:.3f}', flush=True)
      final_loss = running_loss
      running_loss = 0.0

  return final_loss

def default_model_initializer(model_class, model_params):
  model = model_class(*model_params['pos_params'], **model_params['key_params'])
  return model

def builtin_model_initializer(model_class, model_params):
  local_model_params = copy.deepcopy(model_params)
  local_model_params['key_params']['weights'] = local_model_params['key_params']['weights'].DEFAULT
  model = default_model_initializer(model_class, local_model_params)

  if 'finetune_params' in model_params:
    model = define_finetune_model(model, **local_model_params['finetune_params'])

  return model

# training_args:
#   dataset: e.g. torchvision.datasets.CIFAR10
#   transforms_class: e.g. torchvision.models.ResNet18_Weights.DEFAULT.transforms
#   num_epochs: e.g. 1
#   batch_size: e.g. 64
#   model_class: e.g. torchvision.models.resnet18
#   model_params:
#     pos_params: list or tuple positional arguments for model initialization
#     key_params: dictionary keyword arguments for model initialization
#   model_initializer: function that takes in model_class, model_params and returns model instance
#   loss_fn_class:  e.g. torch.nn.CrossEntropyLoss
#   optimizer_class: e.g. torch.optim.Adam
#   optimizer_params: dictionary keyword arguments for optimizer initialization
# NOTE: Do not manually run this with num_dist_proc > 1
def train_process(rank, training_args, num_dist_proc=0):
  print(f"Train process rank {rank}", flush=True)
  set_random_seeds()

  is_distributed = num_dist_proc > 0

  if is_distributed:
    print(f"Running DDP on rank {rank}", flush=True)
    proc_setup(rank, num_dist_proc)
  device = torch.device(f"cuda:{rank}")

  model, train_loader, test_loader = get_model_definition(training_args, is_distributed=is_distributed)

  model.to(device)
  if is_distributed:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

  loss_fn = training_args['loss_fn_class']()
  optimizer_key_params = training_args['optimizer_params']
  optimizer_class = training_args['optimizer_class']
  optimizer = optimizer_class(model.parameters(), **optimizer_key_params)

  num_epochs = training_args['num_epochs']

  losses = []
  accuracies = []
  for epoch in range(1, num_epochs+1):
    loss = train(model, loss_fn, optimizer, device, train_loader, epoch=epoch)
    losses.append(loss)

    if rank == 0:
      acc = test(model, device, test_loader)
      accuracies.append(acc)
      print(f'Accuracy after epoch {epoch}: {100 * acc} %', flush=True)

  if is_distributed:
    cleanup()

def launch(training_args, num_proc=2):
  '''
  if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
  '''
  train_process_args = (copy.deepcopy(training_args), num_proc,)
  mp.spawn(train_process, args=train_process_args, nprocs=num_proc, join=True)

def get_model_definition(training_args, is_distributed=False):
  dataset = training_args['dataset']
  transforms = training_args['transforms']
  batch_size = training_args['batch_size']
  train_loader, test_loader = get_data_loaders(dataset, transforms, batch_size=batch_size, is_distributed=is_distributed)

  model_params = training_args['model_params']
  #model_params['train_loader'] = train_loader
  model_pos_params = model_params['pos_params']
  model_key_params = model_params['key_params']
  model_class = training_args['model_class']
  model_init_fn = default_model_initializer
  if 'model_initializer' in training_args:
    model_init_fn = training_args['model_initializer']
  model = model_init_fn(model_class, model_params)

  return model, train_loader, test_loader

def get_model(training_args):
  local_training_args = copy.deepcopy(training_args)
  model, _, _ = get_model_definition(local_training_args)
  return model

'''
def train(rank, num_epochs, num_proc, ckpt_dir):
    set_random_seeds()
    proc_setup(rank, num_proc)

    print(f"Running DDP on rank {rank}", flush=True)
    device = torch.device(f"cuda:{rank}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    batch_size = 128

    train_set = torchvision.datasets.CIFAR10(root='../data', train=True, download=False, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root='../data', train=False, download=False, transform=transform)

    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset=train_set)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, sampler=train_sampler, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    conv_net = ConvNet().to(device)

#     base_net = conv_net

#     # Register hook to get activations for each layer of convnet
#     activations = {}
#     def get_activations(name):
#         def hook(model, input, output):
#             activations[name] = torch.flatten(output.detach(), start_dim=1)
#         return hook

#     for name, module in base_net.named_modules():
#         module.register_forward_hook(get_activations(name))

    # Train base network
    base_net = DDP(conv_net, device_ids=[rank])

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(base_net.parameters(), lr=1e-3)

    best_accuracy = -float('inf')
    for epoch in range(num_epochs):

        base_net.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = base_net(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            samples = 2000
            if i % samples == samples-1:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / samples:.3f}', flush=True)
                running_loss = 0.0

        if rank == 0:
          accuracy = test(model=base_net, device=device, test_loader=test_loader)
          if accuracy > best_accuracy:
            torch.save(base_net.state_dict(), f"{ckpt_dir}/conv_net_{epoch}.pth")
            best_accuracy = accuracy
          print(f'Accuracy of the network on the 10000 test images: {100 * accuracy} %', flush=True)

    cleanup()
'''
