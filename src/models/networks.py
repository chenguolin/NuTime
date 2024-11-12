import math
import random
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from src.models.layers import *


class WinT(nn.Module):
    """1D ViT with sliding windows for time series classification.

    Referred from https://github.com/lucidrains/vit-pytorch and https://github.com/rwightman/pytorch-image-models.
    """
    def __init__(self, config, series_size: int, in_channels: int, window_emb_dim: int, out_channels: int,
                 window_size=16, stride=16, depth=6, heads=8, head_dim=16, mlp_dim=512, pool_mode='cls',
                 pe_mode='learnable', max_num_tokens=1024, dropout=0., pos_dropout=0., attn_dropout=0.,
                 path_dropout=0., init_values=None):
        super(WinT, self).__init__()

        self.features = None
        self.pool_mode, self.pe_mode = pool_mode, pe_mode
        self.num_windows = series_size
        self.transformer_mask_type = config.transformer_mask_type

        self.window_slide = (config.encoder != 'wne')
        # sliding windows for other encoding methods
        if self.window_slide:
            embed_dim = window_emb_dim
            stride = stride
            self.window_embedding = nn.Conv1d(1, embed_dim, kernel_size=window_size, stride=stride)
            self.num_windows = int((series_size - window_size) / stride) + 1
            self.window_embed_norm = nn.LayerNorm(embed_dim)
            self.channel_embedding = nn.Sequential(nn.Linear(in_channels*embed_dim, embed_dim), nn.LayerNorm(embed_dim)) if in_channels > 1 else nn.Identity()
            in_channels = embed_dim

        # mask token
        self.mask_token_num = int(config.transformer_mask_scale * self.num_windows)
        if self.transformer_mask_type == 'learnable':
            self.mask_token = nn.Parameter(torch.zeros(1, 1, in_channels))
            trunc_normal_(self.mask_token, std=0.02)

        # cls_token
        num_tokens = self.num_windows + 1 if pool_mode == 'cls' else self.num_windows
        self.cls_token = nn.Parameter(torch.empty(1, 1, in_channels)) if pool_mode == 'cls' else None
        if self.cls_token is not None: nn.init.normal_(self.cls_token, std=1e-6)  # std=1e-6/0.02 in ViT/DeiT

        # position emb
        if pe_mode == 'learnable':
            self.pos_embed = nn.Parameter(torch.empty(1, num_tokens, in_channels))
            trunc_normal_(self.pos_embed, std=0.02)  # std=0.02 in ViT
        elif pe_mode == 'fixed':
            assert series_size // window_size <= max_num_tokens, 'Parameter `series_size` must be less than or equal to `max_num_tokens`x`window_size`!'
            position = torch.arange(max_num_tokens).unsqueeze(1)  # (max_len, 1)
            div_term = torch.exp(torch.arange(0, in_channels, 2).float() * (-math.log(10000.) / in_channels))  # (hidden_size/2,)
            pos_embed = torch.zeros(1, max_num_tokens, in_channels)
            pos_embed[0, :, 0::2] = torch.sin(position * div_term)  # (max_len, hidden_size/2)
            pos_embed[0, :, 1::2] = torch.cos(position * div_term)  # (max_len, hidden_size/2)
            self.register_buffer('pos_embed', pos_embed)
        else:
            pass
        self.pos_drop = nn.Dropout(pos_dropout)

        self.transformer = Transformer(in_channels, depth, heads, head_dim, mlp_dim, dropout, attn_dropout, path_dropout, init_values, rotary=(pe_mode=='rotary'))
        self.norm = nn.LayerNorm(in_channels)

        self.fc = nn.Linear(in_channels, out_channels)

    def generate_block_mask(self, num_tokens, mask_tokens):
        """1D block-wise masking for time series.
        Referred from https://github.com/microsoft/unilm/blob/master/beit/masking_generator.py.
        """
        mask = torch.zeros((num_tokens))
        mask_count = 0
        while mask_count < mask_tokens:
            delta = 0
            for attempt in range(10):
                max_mask_tokens = mask_tokens - mask_count
                block_len = random.uniform(0, max_mask_tokens)
                aspect_ratio = random.uniform(0.5, 2)
                l = int(round(math.sqrt(block_len * aspect_ratio)))
                if l < num_tokens:
                    left = random.randint(0, num_tokens - l)
                    num_masked = mask[left: left + l].sum()
                    if 0 < l - num_masked <= max_mask_tokens:
                        for i in range(left, left + l):
                            if mask[i] == 0:
                                mask[i] = 1
                                delta += 1
                if delta > 0:
                    break
            if delta == 0:
                break
            else:
                mask_count += delta

        return mask

    def generate_random_mask(self, num_tokens, mask_tokens):
        """1D block-wise masking for time series.
        Referred from https://github.com/pengzhiliang/MAE-pytorch/blob/main/masking_generator.py.
        """
        mask = np.hstack([
            np.zeros(num_tokens - mask_tokens),
            np.ones(mask_tokens),
        ])
        np.random.shuffle(mask)
        if self.cls_token is not None:
            mask = np.insert(mask, 0, 0)

        return torch.Tensor(mask).to(torch.bool)

    def forward(self, x):
        B, C, L = x.shape
        N = self.num_windows

        if self.window_slide:
            x = x.view(-1, L).unsqueeze(1)  # (b*c, 1, l)
            x_embed = self.window_embedding(x).transpose(1, 2).view(B, C, N, -1)  # (b*c, 1, l) -> (b*c, d_, n) -> (b, c, n, d_)
            x_embed = self.window_embed_norm(x_embed)
            x_embed = x_embed.transpose(1, 2).contiguous().view(B, N, -1)  # (b, c, n, d_) -> (b, n, c, d_) -> (b, n, c*d_)
            x_embed = self.channel_embedding(x_embed)  # (b, n, c*d_) -> (b, n, d)
        else:
            x_embed = x.transpose(1, 2)
        
        # learnable mask
        if self.transformer_mask_type == 'learnable':
            bool_masked_pos = self.generate_block_mask(num_tokens=N, mask_tokens=self.mask_token_num)
            mask_token = self.mask_token.expand(B, N, -1)
            w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
            x_embed = x_embed * (1 - w) + mask_token * w

        # concat cls token
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(B, -1, -1)  # (b, 1, d)
            x_embed = torch.cat((cls_tokens, x_embed), dim=1)

        # add position emb
        x_embed += self.pos_embed[:, :x_embed.shape[1], :]
        x_embed = self.pos_drop(x_embed)

        # random drop mask
        if self.transformer_mask_type == 'drop':
            mask = self.generate_random_mask(num_tokens=N, mask_tokens=self.mask_token_num)
            x_embed = x_embed[:, ~mask].reshape(B, -1, C) 

        output = self.transformer(x_embed)
        output_norm = self.norm(output)

        # cls token emb
        self.features = output_norm[:, 0]

        return self.fc(self.features)


class TCN(nn.Module):
    """TCN model for time series classification.

    Referenced from https://github.com/locuslab/TCN.
    """
    def __init__(self, in_channels: int, out_channels: int, hidden_size=64, kernel_sizes=5, num_blocks=6, dropout=0.):
        super(TCN, self).__init__()

        if type(hidden_size) is int: hidden_size = [hidden_size] * num_blocks
        if type(kernel_sizes) is int: kernel_size = [kernel_sizes] * num_blocks
        assert len(hidden_size) == num_blocks, 'Parameter `hidden_size` must be a list of length `num_blocks`!'
        assert len(kernel_size) == num_blocks, 'Parameter `kernel_size` must be a list of length `num_blocks`!'

        hidden_size = [in_channels] + hidden_size
        layers = []
        for i in range(num_blocks):
            dilation = 2**i
            layers.append(TemporalBlock(
                hidden_size[i], hidden_size[i+1], kernel_size[i], dilation, stride=1, dropout=dropout
            ))
        layers += [
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(1)
        ]

        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_size[-1], out_channels)
        self.features = None  # input of fc layer

    def forward(self, x):
        self.features = self.layers(x)

        return self.fc(self.features)


class InceptionTime(nn.Module):
    """InceptionTime model for time series classification.

    Referred from https://arxiv.org/abs/1909.04939 and https://github.com/hfawaz/InceptionTime.
    """
    def __init__(self, in_channels: int, out_channels: int, norm='bn', hidden_size=32, num_blocks=6, use_residual=True, **kwargs):
        super(InceptionTime, self).__init__()

        self.use_residual, self.num_blocks, self.norm = use_residual, num_blocks, norm
        self.inception, self.shortcut_conv, self.shortcut_norm = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()

        for i in range(num_blocks):
            self.inception.append(InceptionBlock(in_channels if i==0 else hidden_size*4, hidden_size, norm, **kwargs))
            if use_residual and i%3 == 2:
                in_channels_res, channels_res = in_channels if i==2 else hidden_size*4, hidden_size*4
                self.shortcut_conv.append(nn.Conv1d(in_channels_res, channels_res, 1, bias=False))
                self.shortcut_norm.append(nn.BatchNorm1d(channels_res) if self.norm == "bn" else nn.LayerNorm(channels_res))
                # self.shortcut.append(nn.Sequential(
                #     nn.Conv1d(in_channels_res, channels_res, 1, bias=False),
                #     nn.BatchNorm1d(channels_res) if self.norm == "bn" else nn.LayerNorm(channels_res),
                # ))  # instead of `nn.Identity()` when `in_channels_res == channels_res` to be consistent with the official implementation

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten(1)
        self.fc = nn.Linear(hidden_size*4, out_channels)
        self.features = None  # input of fc layer

    def forward(self, x):
        residual = x
        for i in range(self.num_blocks):
            x = self.inception[i](x)
            if self.use_residual and i%3 == 2:
                residual = self.shortcut_conv[i//3](residual)
                if self.norm == 'bn':
                    residual = self.shortcut_norm[i//3](residual)
                else:
                    residual = self.shortcut_norm[i//3](residual.transpose(1, 2)).transpose(1, 2)
                x = x + residual
                # x = x + self.shortcut[i//3](residual)
                residual = x = F.relu(x)
        self.features = self.flatten(self.gap(x))

        return self.fc(self.features)


class ResNetBaseline(nn.Module):
    """ResNet baseline model for time series classificaiton.

    Referred from https://arxiv.org/abs/1611.06455 and https://github.com/cauchyturing/UCR_Time_Series_Classification_Deep_Learning_Baseline.
    """
    def __init__(self, in_channels: int, out_channels: int, hidden_size=64):
        super(ResNetBaseline, self).__init__()

        self.layers = nn.Sequential(
            ResNetBaselineBlock(in_channels, hidden_size),
            ResNetBaselineBlock(hidden_size, hidden_size*2),
            ResNetBaselineBlock(hidden_size*2, hidden_size*2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(1)
        )
        self.fc = nn.Linear(hidden_size*2, out_channels)
        self.features = None  # input of fc layer

    def forward(self, x):
        self.features = self.layers(x)

        return self.fc(self.features)


class FCNBaseline(nn.Module):
    """FCN baseline model for time series classificaiton.

    Referred from https://arxiv.org/abs/1611.06455 and https://github.com/cauchyturing/UCR_Time_Series_Classification_Deep_Learning_Baseline.
    """
    def __init__(self, in_channels: int, out_channels: int, hidden_size=128):
        super(FCNBaseline, self).__init__()

        self.layers = nn.Sequential(
            Conv1dSamePadding(in_channels, hidden_size, 8, stride=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            Conv1dSamePadding(hidden_size, hidden_size*2, 5, stride=1),
            nn.BatchNorm1d(hidden_size*2),
            nn.ReLU(),
            Conv1dSamePadding(hidden_size*2, hidden_size, 3, stride=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(1)
        )
        self.fc = nn.Linear(hidden_size, out_channels)
        self.features = None  # input of fc layer

    def forward(self, x):
        self.features = self.layers(x)

        return self.fc(self.features)


class MLPBaseline(nn.Module):
    """MLP baseline model for time series classificaiton.

    Referred from https://arxiv.org/abs/1611.06455 and https://github.com/cauchyturing/UCR_Time_Series_Classification_Deep_Learning_Baseline.
    """
    def __init__(self, in_channels: int, out_channels: int, hidden_size=512):
        super(MLPBaseline, self).__init__()

        self.layers = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Dropout(0.1),
            nn.Linear(in_channels, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.fc = nn.Linear(hidden_size, out_channels)
        self.features = None  # input of fc layer

    def forward(self, x):
        self.features = self.layers(x)

        return self.fc(self.features)


class GRUBaseline(nn.Module):
    """GRU baseline model for time series classificaiton.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 hidden_size=512, layer_num=2):
        super(GRUBaseline, self).__init__()

        self.cell = nn.GRU(input_size=in_channels, hidden_size=hidden_size,
                           num_layers=layer_num, dropout=0.0, batch_first=True)
        self.fc = nn.Linear(hidden_size, out_channels)
        self.features = None

    def forward(self, x):
        B, C, L = x.shape
        gru_input = x.transpose(1, 2).contiguous() # (b, l, c)
        gru_output, hn = self.cell(gru_input)
        self.features = hn[-1]
        fc_output = self.fc(self.features) # use last hidden state

        return fc_output
