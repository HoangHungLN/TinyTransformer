class TinyConfig:
  def __init__(self):
    #Hardware constrants
    self.embed_dim = 64
    self.num_heads = 2
    self.depth = 3
    self.ffn_mul = 2
    self.num_classes = 10
    self.max_seq_len = 64

    #Image params
    self.img_size = 64
    self.patch_size = 8
    self.in_channels = 3

    #Audio params
    self.audio_features = 40
    self.max_audio_len = 64
