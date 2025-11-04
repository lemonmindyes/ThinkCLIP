import numpy as np
import torch
import torch.nn as nn

from config import Config
from gpt import GPT
from vit import VIT


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


class ThinkCLIP(nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        # base param
        self.align_dim = config.align_dim
        self.img_dim = config.img_dim
        self.text_dim = config.text_dim
        # base module
        self.img_encoder = VIT(config)
        self.text_encoder = GPT(config)
        self.img_norm = RMSNorm(self.img_dim)
        self.text_norm = RMSNorm(self.text_dim)
        self.to_img_out = nn.Linear(self.img_dim, self.align_dim)
        self.to_text_out = nn.Linear(self.text_dim, self.align_dim)
        self.scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_img(self, img):
        # img:[b, c, h, w]
        x = self.img_encoder(img)
        x = self.img_norm(x)
        x = self.to_img_out(x)
        return x

    def encode_text(self, text, attn_mask, eos_token_id):
        # text:[b, n]
        x = self.text_encoder(text, attn_mask)
        x = self.text_norm(x)
        # get eos token
        eos_mask = (text == eos_token_id)
        eos_idx = eos_mask.float().argmax(dim = -1)
        x = x[torch.arange(x.shape[0]), eos_idx]
        x = self.to_text_out(x)
        return x

    def forward(self, img, text, attn_mask, eos_token_id):
        # img:[b, c, h, w]
        # text:[b, n]
        img_feature = self.encode_img(img)
        text_feature = self.encode_text(text, attn_mask, eos_token_id)
        # normalize feature
        img_feature = img_feature / img_feature.norm(dim = -1, keepdim = True)
        text_feature = text_feature / text_feature.norm(dim = -1, keepdim = True)
        # cosine similarity
        scale = torch.exp(self.scale)
        logit_img = scale * torch.matmul(img_feature, text_feature.t())
        logit_text = logit_img.t()
        return logit_img, logit_text


if __name__ == "__main__":
    config = Config()
    model = CLIP(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(total_params)
    print(model)
    img = torch.randn(2, 3, 224, 224)
    text = torch.randint(0, 6400, (2, 64))
    # text[0, 22] = 2
    # text[1, 33] = 2
    # text[0, 23:] = 0
    # text[1, 34:] = 0
    # print(text)
    logit_img, logit_text = model(img, text, attn_mask = None, eos_token_id = 2)
    print(logit_img)
    print(logit_text)