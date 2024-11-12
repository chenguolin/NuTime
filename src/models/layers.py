import functools

import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import DropPath


def get_relative_position_index(h, w, use_cls=False):
    """Get (2D) relative position for attention map by indexing a learnable embedding.

    Referred from https://github.com/microsoft/Swin-Transformer.
    """
    coords = torch.stack(torch.meshgrid([torch.arange(h), torch.arange(w)]))  # (2, h, w)
    coords_flatten = torch.flatten(coords, 1)  # (2, h*w)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, h*w, h*w)
    # shift to start from 0
    relative_coords[0] += h - 1
    relative_coords[1] += w - 1
    relative_coords[0] *= 2*w - 1
    index = relative_coords.sum(0)  # (h*w, h*w)

    num = (2*h - 1) * (2*w - 1)
    if use_cls:  # use class token
        num += 3  # 3: (cls to token) & (token to cls) & (cls to cls)
        index = F.pad(index, (1, 0, 1, 0))  # (h*w+1, h*w+1)
        index[0, 0:], index[0:, 0], index[0, 0] = num-3, num-2, num-1

    return index

class LayerScale(nn.Module):
    """Layer scale for Transformers.

    Referred from https://github.com/rwightman/pytorch-image-models.
    """
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class MLPMixer(nn.Module):
    """MLPMixer model for time series classificaiton.

    Referred from https://github.com/lucidrains/mlp-mixer-pytorch.
    """
    def __init__(self, dim, depth, tokens, token_dim, channel_dim, dropout=0., path_dropout=0., init_values=None):
        super(MLPMixer, self).__init__()

        self.layers = nn.ModuleList()
        dpr = [rate.item() for rate in torch.linspace(0, path_dropout, depth)]  # stochastic depth decay rule
        chan_first, chan_last = functools.partial(nn.Conv1d, kernel_size=1), nn.Linear
        for i in range(depth):
            self.layers.extend([
                PreNorm(dim, FeedForward(tokens, token_dim, dropout, dense_layer=chan_first)),  # default token_dim=dim/2
                LayerScale(dim, init_values=init_values) if init_values else nn.Identity(),
                DropPath(dpr[i]) if dpr[i] > 0. else nn.Identity(),  # token-wise FeedForward: e.g., (B,N,D)->(B,N',D)->(B,N,D)

                PreNorm(dim, FeedForward(dim, channel_dim, dropout, dense_layer=chan_last)),  # default channel_dim=dim*4
                LayerScale(dim, init_values=init_values) if init_values else nn.Identity(),
                DropPath(dpr[i]) if dpr[i] > 0. else nn.Identity()  # channel-wise FeedForward: e.g., (B,N,D)->(B,N,D')->(B,N,D)
            ])

    def forward(self, x):
        for i in range(len(self.layers)//6):
            ff_t, ls1, dp1, ff_c, ls2, dp2 = self.layers[i*6:i*6+6]
            x = dp1(ls1(ff_t(x))) + x  # token mixing
            x = dp2(ls2(ff_c(x)))  # channel mixing

        return x


class Transformer(nn.Module):
    """Transformer model for time series classification.

    Referred from https://github.com/lucidrains/vit-pytorch.
    """
    def __init__(self, dim, depth, heads, head_dim, mlp_dim, dropout=0., attn_dropout=0., path_dropout=0., init_values=None, use_cls=False, rotary=False):
        super(Transformer, self).__init__()

        self.layers = nn.ModuleList()
        dpr = [rate.item() for rate in torch.linspace(0, path_dropout, depth)]  # stochastic depth decay rule
        for i in range(depth):
            self.layers.extend([
                PreNorm(dim, Attention(dim, heads, head_dim, dropout, attn_dropout, use_cls=use_cls, rotary=rotary)),
                LayerScale(dim, init_values=init_values) if init_values else nn.Identity(),
                DropPath(dpr[i]) if dpr[i] > 0. else nn.Identity(),  # Attention

                PreNorm(dim, FeedForward(dim, mlp_dim, dropout)),
                LayerScale(dim, init_values=init_values) if init_values else nn.Identity(),
                DropPath(dpr[i]) if dpr[i] > 0. else nn.Identity(),  # FeedForward
            ])

    def forward(self, x, attn_mask=None, relative_pos_embed=None):
        for i in range(len(self.layers)//6):
            attn, ls1, dp1, ff, ls2, dp2 = self.layers[i*6:i*6+6]
            x = dp1(ls1(attn(x, attn_mask=attn_mask, relative_pos_embed=relative_pos_embed))) + x
            x = dp2(ls2(ff(x))) + x

        return x


class Attention(nn.Module):
    """Attention layer for Transformer.

    Referred from https://github.com/lucidrains/vit-pytorch and https://github.com/rwightman/pytorch-image-models.
    """
    def __init__(self, dim, heads=8, head_dim=16, dropout=0., attn_dropout=0., project_out=True, use_cls=False, rotary=False):
        super(Attention, self).__init__()
        assert dim % heads == 0, 'Parameter `dim` must be divisible by parameter `heads`'

        inner_dim = head_dim * heads  # i.e., attention dimensionality
        project_out = project_out if (inner_dim == dim and heads == 1) else True

        self.heads, self.head_dim = heads, head_dim
        self.scale = head_dim ** -0.5
        self.use_cls = use_cls
        self.rotary = rotary

        self.to_qkv = nn.Linear(dim, inner_dim*3, bias=False)
        self.attend = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(attn_dropout)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, attn_mask=None, relative_pos_embed=None):
        (b, n, _), h, hd = x.shape, self.heads, self.head_dim
        qkv = self.to_qkv(x).reshape(b, n, 3, h, hd).permute(2, 0, 3, 1, 4)  # (b, n, 3*h*hd) -> (3, b, h, n, hd)
        q, k, v = qkv.unbind(dim=0)  # 3 * (b, h, n, hd)

        if relative_pos_embed is not None:
            if self.rotary:
                cos, sin = torch.cos(relative_pos_embed[:, :, :x.shape[1], :]), torch.sin(relative_pos_embed[:, :, :x.shape[1], :])
                # rotate q
                q1, q2 = q[..., :q.shape[-1]//2], q[..., q.shape[-1]//2:]
                rotate_half_q = torch.cat((-q2, q1), dim=-1)
                q = q * cos + rotate_half_q * sin
                # rotate k
                k1, k2 = k[..., :k.shape[-1]//2], k[..., k.shape[-1]//2:]
                rotate_half_k = torch.cat((-k2, k1), dim=-1)
                k = k * cos + rotate_half_k * sin
            else:  # relative position bias
                num_windows = x.shape[1]
                if self.use_cls: num_windows -= 1
                index = get_relative_position_index(num_windows, 1, self.use_cls)
                bias = relative_pos_embed[index.view(-1)].view(x.shape[1], x.shape[1], -1)  # (n, n, h)
                bias = bias.permute(2, 0, 1).contiguous().unsqueeze(0)  # (1, h, n, n)

        dots = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (b, h, n, n)
        dots += bias if (relative_pos_embed is not None and not self.rotary) else 0.  # relative position bias
        if attn_mask is not None: dots.masked_fill_(attn_mask, -1e9)

        attn = self.attend(dots)
        attn = self.attn_drop(attn)

        output = torch.matmul(attn, v).transpose(1, 2).reshape(b, n, h*hd)  # (b, h, n, hd) -> (b, n, h*hd)

        return self.to_out(output)  # (b, n, d)


class FeedForward(nn.Module):
    """Feed-forward layer for Transformer.

    Referenced from https://github.com/lucidrains/vit-pytorch.
    """
    def __init__(self, dim, hidden_dim, dropout=0., dense_layer=nn.Linear):
        super(FeedForward, self).__init__()

        self.layers = nn.Sequential(
            dense_layer(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            dense_layer(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.layers(x)


class PreNorm(nn.Module):
    """Pre-normalization layer for Transformer.

    Referenced from https://github.com/lucidrains/vit-pytorch.
    """
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class TemporalBlock(nn.Module):
    """Basic block in TCN, composed sequentially of two causal convlutions (with ReLU) and a skip connection.

    Referenced from https://github.com/locuslab/TCN.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, stride=1, dropout=0.):
        super(TemporalBlock, self).__init__()

        padding = (kernel_size - 1) * dilation
        self.layers = nn.Sequential(
            nn.utils.weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation)),
            Chomp1d(padding), nn.ReLU(), nn.Dropout(dropout),
            nn.utils.weight_norm(nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding, dilation)),
            Chomp1d(padding), nn.ReLU(), nn.Dropout(dropout)
        )

        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

        self._init_weights([self.layers, self.shortcut])

    def forward(self, x):
        output = self.layers(x)
        output += self.shortcut(x)
        return F.relu(output)

    def _init_weights(self, net_list):
        if not isinstance(net_list, list):
            net_list = [net_list]

        for net in net_list:
            for m in net.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)


class Chomp1d(nn.Module):
    """Remove the last `chomp_size` samples of the time dimension of `x`.

    Referenced from https://github.com/locuslab/TCN.
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class InceptionBlock(nn.Module):
    """InceptionTime block for time series classification.

    Referred from https://arxiv.org/abs/1909.04939 and https://github.com/hfawaz/InceptionTime.
    """
    def __init__(self, in_channels: int, out_channels: int, norm='bn', bottleneck_channels=32, kernel_size=40, use_bottleneck=True):
        super(InceptionBlock, self).__init__()

        kernel_sizes = [kernel_size//(2**i) for i in range(3)]
        # kernel_sizes = [10 for i in range(3)]
        use_bottleneck = use_bottleneck if in_channels>1 else False
        self.norm_mode = norm

        self.bottleneck = Conv1dSamePadding(in_channels, bottleneck_channels, kernel_size=1, bias=False) if use_bottleneck else nn.Identity()
        self.convs = nn.ModuleList([
            Conv1dSamePadding(bottleneck_channels if use_bottleneck else in_channels, out_channels, kernel_sizes[i], bias=False) for i in range(3)
        ])
        self.maxpool = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            Conv1dSamePadding(in_channels, out_channels, kernel_size=1, bias=False)
        )
        self.norm = nn.BatchNorm1d(out_channels*4) if norm == 'bn' else nn.LayerNorm(out_channels*4)

    def forward(self, x):
        bottleneck_output = self.bottleneck(x)
        output = torch.cat([self.convs[i](bottleneck_output) for i in range(3)] + [self.maxpool(x)], dim=1)
        if self.norm_mode == 'bn':
            output = self.norm(output)
        else:
            output = self.norm(output.transpose(1, 2)).transpose(1, 2)

        return F.relu(output)


class ResNetBaselineBlock(nn.Module):
    """Basic block for time series classification ResNet baseline.

    Referred from https://arxiv.org/abs/1611.06455 and https://github.com/cauchyturing/UCR_Time_Series_Classification_Deep_Learning_Baseline.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super(ResNetBaselineBlock, self).__init__()

        channels = [in_channels] + [out_channels] * 3
        kernel_size = [8, 5, 3]

        self.layers = []
        for i in range(len(kernel_size)):
            self.layers.append(
                nn.Sequential(
                    Conv1dSamePadding(channels[i], channels[i+1], kernel_size[i], stride=1),
                    nn.BatchNorm1d(channels[i+1]),
                    nn.ReLU() if i < len(kernel_size)-1 else nn.Identity(),
                )
            )
        self.layers = nn.Sequential(*self.layers)

        self.shortcut = nn.Sequential(
            Conv1dSamePadding(in_channels, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm1d(out_channels)
        ) if in_channels != out_channels else nn.BatchNorm1d(out_channels)
        # instead of `nn.Identity()` when `in_channels == out_channels` to be consistent with the official implementation

    def forward(self, x):
        output = self.layers(x)
        output += self.shortcut(x)

        return F.relu(output)


class Conv1dSamePadding(nn.Module):
    """Conv1d with `same` padding for PyTorch.

    Referred from https://github.com/pytorch/pytorch/issues/3867.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, **kwargs):
        super(Conv1dSamePadding, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride, dilation=dilation, **kwargs)

    def forward(self, x):
        len_out = len_in = x.shape[-1]  # input length
        len_pad = (len_out-1) * self.stride + (self.kernel_size-1) * self.dilation + 1 - len_in
        padding = (len_pad//2, len_pad - len_pad//2)

        return self.conv1d(F.pad(x, padding, 'constant', 0))
