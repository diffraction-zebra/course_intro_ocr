import torch
from torch import nn

class MyViT(nn.Module):
  def __init__(self, patch_kernel, dim, layers, heads, ffn_dim, 
                n_classes, norm_out):
    super(MyViT, self).__init__()

    assert 64 % patch_kernel == 0
    
    self.norm_out = norm_out

    self.patching = nn.Unfold((patch_kernel, patch_kernel), stride=patch_kernel//2)
    self.projection = nn.Linear(patch_kernel ** 2, dim)

    self.tag_token = nn.Parameter(torch.zeros(dim))
    split_size = 64 // patch_kernel
    n_pos_embedding_tokens = (split_size + (split_size -1))**2
    self.pos_embedding = nn.Parameter((torch.rand(n_pos_embedding_tokens, dim)) - 0.5)

    layer = nn.TransformerEncoderLayer(dim, heads, ffn_dim,
                                       batch_first=True, norm_first=True)
    self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
    self.out_layer = nn.Linear(dim, n_classes)

  def forward(self, x):
    x = self.patching(x) # [batch : kernel * kernel : tokens]
    x = torch.movedim(x, 1, 2)
    tokens = self.pos_embedding + self.projection(x) # [batch : tokens : dim]

    tag_tokens = torch.stack([self.tag_token] * len(tokens), dim=0)
    tag_tokens = torch.unsqueeze(tag_tokens, 1) # [batch : 1 : dim]
    tokens = torch.cat([tag_tokens, tokens], 1)

    features = self.encoder(tokens)[:, 0]
    weights = self.out_layer.weight.T
    if self.norm_out:
      features = nn.functional.normalize(features, dim=-1)
      weights = nn.functional.normalize(weights, dim=-1)

    out = torch.matmul(features, weights)

    return out
