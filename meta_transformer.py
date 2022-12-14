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
    self.num_classes = dict(self._base_model.named_modules())[layer_names[-1]].out_features
    self.fc = torch.nn.Linear(self._projection_dim, self.num_classes)

    #self.templates = torch.zeros((self.num_classes, self.num_classes), requires_grad=False, device=input_batch.device)

  def _activations_hook(self, name):

    def hook(model, input, output):
      self._activations[name] = torch.flatten(output.detach(), start_dim=1)

    return hook

  def _get_batch_activations(self, inputs):

    self._base_model.eval()
    _ = self._base_model(inputs)

    return self._activations

  def _get_activation_projections(self, activations):
    activation_projections = [self.embeddings[name](activations[name]) for name in self._layer_names]

    return activation_projections

  def _forward(self, activations):
    activation_projections = self._get_activation_projections(activations)
    activation_projections = torch.stack(activation_projections, dim=1)
    output = self.transformer(activation_projections)
    output = self.fc(output)[:, -1, :]
    return output

  def forward(self, inputs):
    activations = self._get_batch_activations(inputs)
    fc_name = self._layer_names[-1]
    output = torch.zeros_like(activations[fc_name])
    if self.training:
      output = self._forward(activations)

      """
      inds = torch.max(output, 1).indices.unsqueeze(1).repeat((1, self.num_classes))
      for class_idx in range(self.num_classes):
        batch_templates = torch.reshape(activations[fc_name][inds == class_idx], (-1, self.num_classes))
        if batch_templates.shape[0] > 0:
          avg_template = torch.mean(batch_templates, 0, keepdim=True)
          self.templates[class_idx, :] = 0.9 * self.templates[class_idx, :] + 0.1 * avg_template
      #print(self.templates, flush=True)
      """

    else:
      output = self._forward(activations)
      """
      fc_activations = activations[fc_name]
      max_vals, max_inds = torch.max(fc_activations, 1)
      max_inds = torch.unsqueeze(max_inds, 1).repeat((1, self.num_classes))
      max_vals = torch.unsqueeze(max_vals, 1)
      for class_idx in range(self.num_classes):
        template = self.templates[class_idx, :].squeeze()
        fc_activations[max_inds == class_idx] = template.repeat((fc_activations[max_inds == class_idx].shape[0] // template.shape[0]))
      activations[fc_name] = fc_activations
      output = self._forward(activations)
      """
      """
      batch_size = activations[fc_name].shape[0]
      for class_idx in range(self.num_classes):
        activations[fc_name] = self.templates[class_idx, :].repeat((batch_size, 1))
        primed_output = self._forward(activations)
        #output[:, i] = torch.max(primed_output, 1).values
        output[:, class_idx] = primed_output[:, class_idx]
      """

    return output
