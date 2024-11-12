import math
import torch
from torch import nn


def cos_sin(x):
    return torch.cat([torch.cos(x), torch.sin(x)], -1)


class PeriodicEncoder(nn.Module):
    """Periodic activation functions to encode numeric feature to embedding.

    Referred from https://arxiv.org/abs/2203.05556 and https://github.com/Yura52/tabular-dl-num-embeddings.
    """
    def __init__(self, config):
        super().__init__()

        feature = config.num_channels
        dim = config.period_dim
        sigma = config.period_sigma
        init_mode = config.period_init
        trainable = config.period_train

        self.out_channels = dim * 2
        if init_mode == 'log-linear':
            coefficients = sigma ** (torch.arange(dim) / dim)
            coefficients = coefficients[None].repeat(feature, 1) # (config.num_channels, dim)
        else:
            assert init_mode == 'normal'
            coefficients = torch.normal(0.0, sigma, (feature, dim))
        if trainable:
            self.coefficients = nn.Parameter(coefficients)  # type: ignore[code]
        else:
            self.register_buffer('coefficients', coefficients)

    def forward(self, x: torch.Tensor):
        x = x.transpose(1, 2)
        enc_x = cos_sin(2 * math.pi * self.coefficients * x)
        return enc_x.transpose(1, 2).contiguous()
