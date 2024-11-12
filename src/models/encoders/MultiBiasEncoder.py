import torch
from torch import nn
from timm.models.layers import trunc_normal_


class MultiBiasEncoder(nn.Module):
    """Encode unbounded numerical value with multi-scales at multi-locations."""
    def __init__(self, config) -> None:
        super().__init__()
        self.scale_weighted_sum = config.scale_weighted_sum
        self.loc_weighted_sum = config.loc_weighted_sum
        self.use_loc_emb = config.use_loc_emb
        self.out_channels = config.mb_emb_dim
        # multi bias parameter
        ws = nn.Parameter(torch.empty(config.num_bias, self.out_channels))
        bs = nn.Parameter(torch.empty(config.num_bias, self.out_channels))
        nn.init.uniform_(ws, -1, 1)
        nn.init.uniform_(bs, -1, 1)
        if config.multibias_trainable:
            self.ws = ws
            self.bs = bs
        else:
            self.register_buffer('ws', ws)
            self.register_buffer('bs', bs)
        # pre-defined shared scales
        scales = [config.base_bias ** i for i in range(-(config.num_bias-1)//2, (config.num_bias+1)//2)] if config.scales is None else config.scales
        self.register_buffer('scales', torch.tensor(scales))
        print("scales:", self.scales)
        # pre-defined locations
        locs = [0] if config.locs is None else config.locs
        self.register_buffer('locs', torch.tensor(locs))
        print("locs:", self.locs)
        if self.use_loc_emb:
            self.loc_emb = nn.Parameter(torch.empty(self.locs.shape[-1], self.out_channels))
            trunc_normal_(self.loc_emb, std=0.02)
        self.ln = nn.LayerNorm(self.out_channels, elementwise_affine=False)

    def forward(self, x):
        enc_x = x
        # flatten
        if enc_x.ndim > 2:
            enc_x = enc_x.view(enc_x.shape[0], -1)

        eps = 1e-6
        # location weight and scale weight
        loc_dis = enc_x[..., None] - self.locs    # (B, C, K_loc)
        if self.loc_weighted_sum:
            loc_w = 1 / (torch.log(torch.abs(loc_dis) + 1) + eps)    # (B, C, loc_num)
            loc_w = loc_w / loc_w.sum(dim=-1, keepdim=True)
        if self.scale_weighted_sum:
            scale_w = 1 / (torch.abs(torch.log(torch.abs(loc_dis[..., None]) / self.scales + eps)) + eps)    # (B, C, loc_num, scale_num)
            scale_w = scale_w / scale_w.sum(dim=-1, keepdim=True)

        # get multi-location multi-scale emb
        enc_x = loc_dis
        enc_x = enc_x[..., None, None] * self.ws + self.bs * self.scales.unsqueeze(dim=-1)    # (B, C, loc_num, scale_num, D)
        enc_x = self.ln(enc_x)

        # calculate weighted sum of multi scale for each location
        if self.scale_weighted_sum:
            enc_x = enc_x * scale_w[..., None]
        # enc_x = self.ln(enc_x.sum(dim=-2))    # (B, C, loc_num, D)
        enc_x = enc_x.sum(dim=-2)

        # add loc emb
        if self.use_loc_emb:
            enc_x = enc_x + self.loc_emb

        # calculate weighted sum for multi location
        if self.loc_weighted_sum:
            enc_x = enc_x * loc_w[..., None]
        # enc_x = self.ln(enc_x.sum(dim=-2))    # (B, C, D)
        enc_x = enc_x.sum(dim=-2)

        return enc_x


class MultiBiasEncoderWrapper(nn.Module):
    def __init__(self, config):
        super(MultiBiasEncoderWrapper, self).__init__()
        self.multi_bias_encoder = MultiBiasEncoder(config)

    def forward(self, x):
        B, C, L = x.shape
        x_embed = self.multi_bias_encoder(x).view(B, C, L, -1).transpose(2, 3).view(B, -1, L)

        return x_embed
