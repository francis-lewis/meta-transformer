from determined.pytorch import DataLoader, PyTorchTrial

from sairg_utils import get_model

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

    print(training_args)
    self.model = get_model(training_args)
    print(self.model)
    self.model = self.context.wrap_model(self.model)

  def build_training_data_loader(self):
    pass

  def build_validation_data_loader(self):
    pass

  def train_batch(self, batch, epoch_idx, batch_idx):
    pass

  def evaluate_batch(self, batch):
    pass
