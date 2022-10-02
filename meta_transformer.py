import torch

class MetaTransformer(torch.nn.Module):

  def __init__(self, base_model, layer_names, input_batch):
    super().__init__()

    self._base_model = base_model
    for param in self._base_model.parameters():
      param.requires_grad = False

    self._activations = {}
    self._layer_names = layer_names
    for name, module in self._base_model.named_modules():
      if name in self._layer_names:
        module.register_forward_hook(self._activations_hook(name))

    sample_activations = self._get_batch_activation_list(input_batch)
    self.num_feats = max([layer.shape[-1] for layer in sample_activations])

  def _activations_hook(self, name):

    def hook(model, input, output):
      self._activations[name] = output.detach()

    return hook

  def _get_batch_activation_list(self, inputs):

    self._base_model.eval()
    _ = self._base_model(inputs)
    layers = [torch.flatten(self._activations[name], start_dim=1) for name in self._layer_names]

    return layers

  def _get_activation_tensor(self, inputs):
    layers = self._get_batch_activation_list(inputs)
    activation_tensor = torch.zeros((inputs.shape[0], len(layers), self.num_feats))
    for i, layer in enumerate(layers):
      activation_tensor[:, i, :layer.shape[-1]] = layer

    return activation_tensor

  def forward(self, inputs):
    activations = self._get_activation_tensor(inputs)

    return activations
