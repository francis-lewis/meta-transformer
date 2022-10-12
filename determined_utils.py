import copy
import os

import torch
from determined.pytorch import DataLoader, PyTorchTrial

import sairg_utils

class ModelTrial(PyTorchTrial):
  def __init__(self, context):
    self.context = context
    training_arg_keys = [ 'dataset',
                          'transforms',
                          'model_class',
                          'model_params',
                          'model_initializer',
                          'loss_fn_class',
                          'optimizer_class',
                          'optimizer_params',
                          'checkpoint_dir',
                          'checkpoint_prefix',
                         ]
    training_args = {k: self.context.get_hparam(k) for k in training_arg_keys}
    training_args = copy.deepcopy(training_args)
    training_args = sairg_utils.import_arg_classes(training_args)

    model = sairg_utils.get_model(training_args)
    self.model = self.context.wrap_model(model)

    self.loss_fn = training_args['loss_fn_class']()
    optimizer_key_params = training_args['optimizer_params']
    optimizer_class = training_args['optimizer_class']
    optimizer = optimizer_class(self.model.parameters(), **optimizer_key_params)
    self.optimizer = self.context.wrap_optimizer(optimizer)

    self.dataset = training_args['dataset']
    self.transforms = training_args['transforms']

  def build_training_data_loader(self):
    train_set = self.dataset(root='/data', train=True, download=False, transform=self.transforms)
    return DataLoader(train_set, batch_size=self.context.get_per_slot_batch_size(), shuffle=True)

  def build_validation_data_loader(self):
    val_set = self.dataset(root='/data', train=False, download=False, transform=self.transforms)
    return DataLoader(val_set, batch_size=self.context.get_per_slot_batch_size())

  def train_batch(self, batch, epoch_idx, batch_idx):
    data, labels = batch

    self.model.train()
    output = self.model(data)
    training_loss = self.loss_fn(output, labels)

    self.context.backward(training_loss)
    self.context.step_optimizer(self.optimizer)

    return {"training_loss": training_loss}

  def evaluate_batch(self, batch):
    data, labels = batch

    self.model.eval()
    with torch.no_grad():
      output = self.model(data)
      val_loss = self.loss_fn(output, labels)

      preds = output.argmax(dim=1, keepdim=True)
      accuracy = preds.eq(labels.view_as(preds)).sum().item() / len(data)

    return {"validation_loss": val_loss, "accuracy": accuracy}
