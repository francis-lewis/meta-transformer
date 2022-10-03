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
def train_process(rank, input_training_args, num_dist_proc=0):
  print(f"Train process rank {rank}", flush=True)
  set_random_seeds()

  is_distributed = num_dist_proc > 0

  if is_distributed:
    print(f"Running DDP on rank {rank}", flush=True)
    proc_setup(rank, num_dist_proc)
  device = torch.device(f"cuda:{rank}")

  training_args = copy.deepcopy(input_training_args)

  model, train_loader, test_loader = get_model_definition(training_args, is_distributed=is_distributed)

  model.to(device)
  if is_distributed:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

  loss_fn = training_args['loss_fn_class']()
  optimizer_key_params = training_args['optimizer_params']
  optimizer_class = training_args['optimizer_class']
  optimizer = optimizer_class(model.parameters(), **optimizer_key_params)

  checkpoint_dir = training_args['checkpoint_dir']
  checkpoint_prefix = training_args['checkpoint_prefix']
  checkpoint_filename = checkpoint_prefix + ".pth"
  checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)

  num_epochs = training_args['num_epochs']

  losses = []
  accuracies = []
  for epoch in range(1, num_epochs+1):
    loss = train(model, loss_fn, optimizer, device, train_loader, epoch=epoch)
    losses.append(loss)

    if rank == 0:
      accuracy = test(model, device, test_loader)
      if len(accuracies) == 0 or accuracy > max(accuracies):
        print(f'Saving {checkpoint_filename} at epoch {epoch}')
        state_dict = copy.deepcopy(model.state_dict())
        if is_distributed:
          state_dict = copy.deepcopy(model.module.state_dict())
        torch.save(state_dict, checkpoint_path)
      print(f'Accuracy after epoch {epoch}: {100 * accuracy} %', flush=True)
      accuracies.append(accuracy)

  if is_distributed:
    cleanup()

def launch(training_args, num_proc=2):

  checkpoint_dir = training_args['checkpoint_dir']
  if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

  train_process_args = (training_args, num_proc,)
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

def get_model(training_args, train=False):
  local_training_args = copy.deepcopy(training_args)
  model, _, _ = get_model_definition(local_training_args)

  checkpoint_dir = local_training_args['checkpoint_dir']
  checkpoint_prefix = local_training_args['checkpoint_prefix']
  checkpoint_filename = checkpoint_prefix + ".pth"
  checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)

  if not os.path.exists(checkpoint_path) and train:
    launch(training_args)
  # checkpoint_path should exist after training
  if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path))

  return model
