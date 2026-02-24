import torch
import torch.nn as nn

class multiHeadAttention(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.embed_dim = cfg.embed_dim
    self.num_heads = cfg.num_heads
    self.head_dim = cfg.embed_dim // cfg.num_heads
    self.scale = self.head_dim ** -0.5

    # Gom Wq, Wk, Wv vào chung một tensor để tối ưu việc load dữ liệu từ SRAM
    self.qkv = nn.Linear(cfg.embed_dim, cfg.embed_dim * 3, bias = False)
    self.proj = nn.Linear(cfg.embed_dim, cfg.embed_dim)

  def forward(self, x):
    B, N, C = x.shape
    #B: số batch, N: số token(NLP)/patch(Vision)/frame(Audio), C = embed_dim

    #Tính Q, K, V
    #[HW mapping]: Load x từ Buffer, Load W_qkv từ SRAM -> (đưa vào) Systolic Array
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    #[Shape]: [B, N, C(embed_dim)] -(qkv)-> [B, N, embed_dim*3] -(reshape)-> [B, N, 3, num_heads, head_dim]
                                                              # -(permute)-> [3, B, num_heads, N, head_dim]
    q, k, v = qkv[0], qkv[1], qkv[2]
    #[Shape] q|k|v: [B, num_heads, N, head_dim]

    #attn_score = Softmax((Q * K^T)/sqrt(embed_dim))
    #attn = (Q * K^T)/sqrt(embed_dim)
    attn = (q @ k.transpose(-2, -1)) * self.scale
    #[Shape] attn: [B, num_heads, N, N]

    #Đưa qua hàm Softmax
    #[HW mapping]: Hiện thực Softmax với exp(x) xấp xỉ dùng LUT hoặc xấp xỉ đa thức
    attn = attn.softmax(dim = -1)

    #Tính attn_score * V
    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    #[Shape] x: [B, num_heads, N, head_dim] -(transpose)-> [B, N, num_heads, head_dim] -(reshape)-> [B, N, C]
                                                                                    #(C = num_heads*head_dim)

    x = self.proj(x)
    #[Shape] x: [B, N, C(embed_dim)]
    return x

class Mlp(nn.Module):
  """ Feed-Forward Network.
  [HW Mapping]: Tái sử dụng khối MMU đã dùng cho Attention.
  """
  def __init__(self, cfg):
    super().__init__()
    hidden_features = int(cfg.embed_dim * cfg.ffn_mul)

    #[HW Mapping]: lấy Weights từ SRAM và dùng MAC cho fully connected
    #[HW Mapping]: Hiện thực hàm ReLU
    self.fc1 = nn.Linear(cfg.embed_dim, hidden_features)
    self.act = nn.ReLU()
    self.fc2 = nn.Linear(hidden_features, cfg.embed_dim)

  def forward(self, x):
    x = self.fc1(x)
    x = self.act(x)
    x = self.fc2(x)
    return x

class LayerNorm(nn.Module):
  """
  LayerNorm chuẩn.
  [HW Mapping]: Vector ALU.
  Lưu ý: Để tối ưu phần cứng, có thể thay thế phép chia căn bậc hai (sqrt)
  bằng phép dịch bit (bit-shift) nếu ép phương sai về lũy thừa của 2,
  hoặc dùng bảng tra Inverse Square Root (Fast InvSqrt).
  """
  def __init__(self, normalized_shape, eps = 1e-5):
    super().__init__()
    self.weight = nn.Parameter(torch.ones(normalized_shape)) #Gamma
    self.bias = nn.Parameter(torch.zeros(normalized_shape)) #Beta
    self.eps = eps
    self.normalized_shape = normalized_shape

  def forward(self, x):
    u = x.mean(dim = -1, keepdim = True)
    s = (x - u).pow(2).mean(dim = -1, keepdim = True)
    x = (x - u) / torch.sqrt(s + self.eps)
    x = x * self.weight + self.bias

    return x
