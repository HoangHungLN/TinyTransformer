import torch
import torch.nn as nn
from layers import multiHeadAttention, Mlp, LayerNorm

class encoder(nn.Module):
  # Pre-normalization
  def __init__(self, cfg):
    super().__init__()
    self.norm1 = LayerNorm(cfg.embed_dim)
    self.attn = multiHeadAttention(cfg)
    self.norm2 = LayerNorm(cfg.embed_dim)
    self.mlp = Mlp(cfg)

  def forward(self, x):
    x = x + self.attn(self.norm1(x))
    x = x + self.mlp(self.norm2(x))
    return x

class tinyTransformer(nn.Module):
  """
  Hardware: Accelerator Core.
  """
  def __init__(self, cfg):
    super().__init__()
    self.pos_embed = nn.Parameter(torch.zeros(1, cfg.max_seq_len, cfg.embed_dim))
    self.encoders = nn.ModuleList([encoder(cfg) for _ in range(cfg.depth)])
    self.norm = LayerNorm(cfg.embed_dim)
    self.head = nn.Linear(cfg.embed_dim, cfg.num_classes)

  def forward(self, x):
    # Giả sử x có độ dài thay đổi, cắt pos_embed cho khớp
    if x.shape[1] > self.pos_embed.shape[1]:
      x = x[:, :self.pos_embed.shape[1], :]
    # Cộng Positional Embedding
    x = x + self.pos_embed[:, :x.shape[1], :]

    for encoder in self.encoders:
      x = encoder(x)

    x = self.norm(x)
    x = x.mean(dim = 1) # Global Average Pooling
    x = self.head(x)
    return x
