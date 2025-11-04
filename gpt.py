import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from config import Config


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4, f'x.ndim != 4'
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x2 * cos - x1 * sin
    out = torch.cat((y1, y2), dim = -1)
    out = out.to(dtype = x.dtype)
    return out


class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-5, device = None):
        super().__init__()
        # base param
        self.dim = dim
        self.eps = eps
        # base module
        self.weight = nn.Parameter(torch.ones(dim, device = device, dtype = torch.float32))

    def forward(self, x):
        # x:[b, n, d]
        assert x.shape[-1] == self.dim, f'x.shape[-1] != self.dim'
        t, dtype = x.float(), x.dtype
        t = t * torch.rsqrt(torch.mean(t ** 2, dim = -1, keepdim = True) + self.eps)
        return (t * self.weight).to(dtype)


class Attention(nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        # base param
        self.is_flash = config.is_flash
        self.dim = config.text_dim
        self.n_heads = config.text_n_heads
        self.n_kv_heads = config.text_n_kv_heads
        assert self.n_heads % self.n_kv_heads == 0, 'n_heads must be divisible by n_kv_heads'
        self.head_dim = self.dim // self.n_heads
        self.scale = self.head_dim ** -0.5
        self.dropout_rate = config.text_dropout_rate
        # base module
        self.to_q = nn.Linear(self.dim, self.n_heads * self.head_dim, bias = False)
        self.to_kv = nn.Linear(self.dim, 2 * self.n_kv_heads * self.head_dim, bias = False)
        self.to_out = nn.Linear(self.n_heads * self.head_dim, self.dim)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x, cos_sin, attn_mask):
        # x:[b, n, d]
        q = self.to_q(x)
        kv = self.to_kv(x).chunk(2, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b n h d', d = self.head_dim), (q, kv[0], kv[1]))

        # rotary embedding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        # attn
        q, k, v = map(lambda t: rearrange(t, 'b n h d -> b h n d'), (q, k, v))
        enable_gqa = self.n_heads != self.n_kv_heads
        if self.is_flash:
            if attn_mask is not None:
                attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
            out = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask = attn_mask,
                                                             dropout_p = self.dropout_rate, enable_gqa = enable_gqa)
        else:
            if enable_gqa:
                k = repeat(k, 'b h n d -> b (r h) n d', r = self.n_heads // self.n_kv_heads)
                v = repeat(v, 'b h n d -> b (r h) n d', r = self.n_heads // self.n_kv_heads)
            attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            if attn_mask is not None:
                attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
                attn_mask = (1.0 - attn_mask) * -1e9
                attn = attn + attn_mask
            attn = torch.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class FFN(nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        # base param
        self.dim = config.text_dim
        self.dropout_rate = config.text_dropout_rate
        # base module
        self.ffn = nn.Sequential(
            nn.Linear(self.dim, self.dim * 4),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.dim * 4, self.dim),
            nn.Dropout(self.dropout_rate)
        )

    def forward(self, x):
        # x:[b, n, d]
        out = self.ffn(x)
        return out


class Transformer(nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        # base param
        self.dim = config.text_dim
        self.n_layers = config.text_n_layers
        # base module
        self.layers = nn.ModuleList([
            nn.ModuleList([
                RMSNorm(self.dim),
                Attention(config),
                RMSNorm(self.dim),
                FFN(config)
            ]) for _ in range(self.n_layers)
        ])

    def forward(self, x, cos_sin, attn_mask):
        # x:[b, n, d]
        for attn_norm, attn, ffn_norm, ffn in self.layers:
            x = x + attn(attn_norm(x), cos_sin, attn_mask)
            x = x + ffn(ffn_norm(x))
        return x


class GPT(nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        # base param
        self.vocab_size = config.vocab_size
        self.max_seq_len = config.max_seq_len
        self.dim = config.text_dim
        self.n_heads = config.text_n_heads
        assert self.dim % self.n_heads == 0, 'dim must be divisible by n_heads'
        self.head_dim = self.dim // self.n_heads
        # base module
        self.token_embedding = nn.Embedding(self.vocab_size, self.dim)
        self.cos, self.sin = self._precompute_rotary_embedding(self.max_seq_len, self.head_dim)
        self.transformer = Transformer(config)
        self.to_out = nn.Identity()

    def _precompute_rotary_embedding(self, seq_len, head_dim, base = 10000):
        freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype = torch.float32) / head_dim))
        t = torch.arange(seq_len, dtype = torch.float32)
        freqs = torch.outer(t, freqs)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def forward(self, x, attn_mask = None):
        # x:[b, n]
        x = self.token_embedding(x)

        n = x.shape[1]
        cos_sin = self.cos[:, 0:n].to(x.device), self.sin[:, 0:n].to(x.device)
        x = self.transformer(x, cos_sin, attn_mask)

        out = self.to_out(x)
        return out


if __name__ == "__main__":
    config = Config()
    config.is_flash = True
    model = GPT(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(total_params)
    print(model)
    n = config.max_seq_len
    data = torch.randint(0, config.vocab_size, (1, n))
    out = model(data)
    print(out.shape)

    # attn = Attention(config)
    # n = 128
    # data = torch.randn(2, n, config.text_dim)
    # cos_sin = model.cos[:, 0:n], model.sin[:, 0:n]
    # attn(data, cos_sin)


