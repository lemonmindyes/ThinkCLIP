import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from config import Config
from utils import check_tuple


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
        self.dim = config.img_dim
        self.n_heads = config.img_n_heads
        assert self.dim % self.n_heads == 0, 'dim must be divisible by n_heads'
        self.head_dim = self.dim // self.n_heads
        self.scale = self.head_dim ** -0.5
        self.dropout_rate = config.img_dropout_rate
        # base module
        self.to_qkv = nn.Linear(self.dim, 3 * self.n_heads * self.head_dim, bias = False)
        self.to_out = nn.Linear(self.n_heads * self.head_dim, self.dim)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x):
        # x:[b, n, d]
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.n_heads), qkv)
        if self.is_flash:
            out = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask = None, dropout_p = self.dropout_rate,
                                                             is_causal = False)
        else:
            attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn = torch.softmax(attn, dim = -1)
            attn = self.dropout(attn)
            out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class FFN(nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        # base param
        self.dim = config.img_dim
        self.dropout_rate = config.img_dropout_rate
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
        self.dim = config.img_dim
        self.n_layers = config.img_n_layers
        # base module
        self.layers = nn.ModuleList([
            nn.ModuleList([
                RMSNorm(self.dim),
                Attention(config),
                RMSNorm(self.dim),
                FFN(config)
            ]) for _ in range(self.n_layers)
        ])

    def forward(self, x):
        # x:[b, n, d]
        for attn_norm, attn, ffn_norm, ffn in self.layers:
            x = x + attn(attn_norm(x))
            x = x + ffn(ffn_norm(x))
        return x


class VIT(nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        # base param
        self.img_h, self.img_w = check_tuple(config.img_size)
        self.p_h, self.p_w = check_tuple(config.patch_size)
        assert self.img_h % self.p_h == 0 and self.img_w % self.p_w == 0, 'img_size must be divisible by patch_size'
        self.channel = config.channel
        self.patch_dim = self.channel * self.p_h * self.p_w
        self.n_patch = (self.img_h // self.p_h) * (self.img_w // self.p_w)
        self.dim = config.img_dim
        self.num_classes = config.num_classes

        # base module
        self.to_patch = nn.Sequential(
            Rearrange('b c (h p_h) (w p_w) -> b (h w) (c p_h p_w)', p_h = self.p_h, p_w = self.p_w),
            RMSNorm(self.patch_dim),
            nn.Linear(self.patch_dim, self.dim),
            RMSNorm(self.dim)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))
        self.pos_embedding = nn.Parameter(torch.randn((1, self.n_patch + 1, self.dim)))
        self.transformer = Transformer(config)
        if self.num_classes is not None:
            self.to_out = nn.Linear(self.dim, self.num_classes)
        else:
            self.to_out = nn.Linear(self.dim, self.dim)

    def forward(self, x):
        # x:[b, c, h, w]
        b = x.shape[0]
        # patch embedding
        x = self.to_patch(x)

        # cls token
        cls_token = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_token, x), dim = 1)
        x = x + self.pos_embedding

        # transformer
        x = self.transformer(x)

        # to_out
        cls = x[:, 0]
        out = self.to_out(cls)
        return out


if __name__ == "__main__":
    config = Config()
    model = VIT(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(total_params)
    print(model)
    data = torch.randn(1, config.channel, config.img_size, config.img_size)
    out = model(data)
    print(out.shape)