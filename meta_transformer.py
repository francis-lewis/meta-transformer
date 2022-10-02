import torch

class MetaTransformer(torch.nn.Module):

  def __init__(self, base_model, layer_names, input_batch, num_transformer_layers=6, num_heads=8, projection_dim=1000):
    super().__init__()

    self._base_model = base_model
    for param in self._base_model.parameters():
      param.requires_grad = False
    self._base_model.to(input_batch.device)

    self._activations = {}
    self._layer_names = layer_names
    for name, module in self._base_model.named_modules():
      if name in self._layer_names:
        module.register_forward_hook(self._activations_hook(name))

    self._projection_dim = (projection_dim // num_heads) * num_heads
    sample_activations = self._get_batch_activations(input_batch)
    activation_embeddings = {}
    for name, activations in sample_activations.items():
      activation_embeddings[name] = torch.nn.Linear(activations.shape[-1], self._projection_dim)
    self.embeddings = torch.nn.ModuleDict(activation_embeddings)
    encoder_layer = torch.nn.TransformerEncoderLayer(self._projection_dim,
                                                     num_heads,
                                                     activation='gelu',
                                                     batch_first=True)
    self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_transformer_layers)
    num_classes = dict(self._base_model.named_modules())[layer_names[-1]].out_features
    self.fc = torch.nn.Linear(self._projection_dim, num_classes)

  def _activations_hook(self, name):

    def hook(model, input, output):
      self._activations[name] = torch.flatten(output.detach(), start_dim=1)

    return hook

  def _get_batch_activations(self, inputs):

    self._base_model.eval()
    _ = self._base_model(inputs)

    return self._activations

  def _get_activation_projections(self, inputs):
    activations = self._get_batch_activations(inputs)
    activation_projections = [self.embeddings[name](activations[name]) for name in self._layer_names]

    return activation_projections

  def forward(self, inputs):
    activations = self._get_activation_projections(inputs)
    activations = torch.stack(activations, dim=1)
    output = self.transformer(activations)
    output = self.fc(output)[:, -1, :]

    return output
