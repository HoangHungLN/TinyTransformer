import torch
from torch import nn

class PatchEmbed(nn.Module):
  """ Xử lý ảnh 64x64x3 """
  def __init__(self, cfg):
    super().__init__()
    #[HW mapping]: Có thể dùng bộ MAC của Transformer accel hoặc khối Conv2d accel riêng hoặc DMA thông minh.
    self.proj = nn.Conv2d(cfg.in_channels, cfg.embed_dim, cfg.patch_size, cfg.patch_size)

  def forward(self, x):
    #[Shape] x: [B, 3, 64, 64] -> [B, 64, 8, 8] -> Flatten -> [B, 64, 64]
    x = self.proj(x).flatten(2).transpose(1, 2)
    return x

class AudioEmbed(nn.Module):
  """ Xử lý Audio Features (MFCC/Mel) """
  def __init__(self,cfg):
    super().__init__()
    #[HW mapping]: Dùng MAC của Transformer accel
    self.proj = nn.Linear(cfg.audio_features, cfg.embed_dim)

  def forward(self, x):
    #Shape x: [B, Time, Features] -> [B, Time, Embed_Dim]
    x = self.proj(x)
    return x
