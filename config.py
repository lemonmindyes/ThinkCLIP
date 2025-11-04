from dataclasses import dataclass


@dataclass
class Config:
    # common
    eps: float = 1e-5
    is_flash: bool = True
    align_dim: int = 512
    # img encoder
    img_size: int = 224
    patch_size: int = 16
    channel: int = 3
    img_dim: int = 384
    img_n_heads: int = 6
    img_n_layers: int = 12
    num_classes: int = None
    img_dropout_rate: float = 0.1
    # text encoder
    vocab_size: int = 6400
    max_seq_len: int = 64
    text_dim: int = 768
    text_n_heads: int = 12
    text_n_kv_heads: int = 3
    text_n_layers: int = 12
    text_dropout_rate: float = 0.1