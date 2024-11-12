import torch
from torch import nn

from .PeriodicEncoder import PeriodicEncoder
from .PLEEncoder import PLEEncoder
from .MultiBiasEncoder import MultiBiasEncoder


class WindowNormEncoder(nn.Module):
    """Window norm emb and mean/std emb."""
    def __init__(self, config, samples=None, targets=None):
        super(WindowNormEncoder, self).__init__()
        assert config.model_series_size % config.window_size == 0, 'Parameter `series_size` must be divisible by Parameter `window_size`!'

        # sliding window without dilation and overlap
        self.window_size = config.window_size
        self.stride = config.stride
        self.window_num = (config.model_series_size - config.window_size) // config.stride + 1
        self.in_channels = config.num_channels
        self.wne_use_in = config.wne_use_in
        self.wne_use_instance_stats = config.wne_use_instance_stats

        # sliding window with multi-window size without overlap
        # self.window_size = [8, 16, 32, 64]
        self.use_ms_embed = config.use_ms_embed
        self.mean_std_encode = config.mean_std_encode

        embed_dim = config.window_emb_dim
        self.window_embedding = nn.Conv1d(1, embed_dim, kernel_size=self.window_size, stride=self.window_size)
        self.window_embed_norm = nn.LayerNorm(embed_dim)

        if self.use_ms_embed:  # concatenate embedding for injecting instance/window-wise mean and std information
            if config.mean_std_encode == 'mbe':
                self.multi_bias_encoder = MultiBiasEncoder(config)
            elif config.mean_std_encode == 'periodic':
                self.multi_bias_encoder = PeriodicEncoder(config)
            elif config.mean_std_encode == 'ple':
                self.multi_bias_encoder = PLEEncoder(config)
                self.multi_bias_encoder.get_bin_edges(samples, targets)
            ms_emb_dim = 2 * config.mb_emb_dim
            if config.mean_std_encode == 'ple':
                ms_emb_dim = self.multi_bias_encoder.out_channels
            self.ms_embed_fc = nn.Sequential(
                nn.Linear(embed_dim+ms_emb_dim, embed_dim),
                nn.LayerNorm(embed_dim)
            )

        self.use_channel_emb_sum = config.use_channel_emb_sum
        if config.num_channels > 1:
            self.channel_embedding = nn.Sequential(nn.Linear(config.num_channels*embed_dim, embed_dim), nn.LayerNorm(embed_dim))
        else:
            self.channel_embedding = nn.Identity()


    def concat_overlap_windows(self, x):
        indexer = torch.arange(self.window_size)[None, None, :] + self.stride * torch.arange(self.window_num)[:, None]
        out = x[:, :, indexer].view(-1, self.in_channels, self.window_size * self.window_num)

        return out


    def forward(self, x):
        # reconstruct sequence by concat overlap windows
        # x = self.concat_overlap_windows(x)
        B, C, L = x.shape
        N, W = self.window_num, self.window_size

        # only for ablation
        x_in_mean = x.mean(dim=-1, keepdim=True).detach().expand(B, C, N).unsqueeze(-1)
        x_in_std = x.std(dim=-1, keepdim=True).detach().expand(B, C, N).unsqueeze(-1)
        # window norm
        x = x.view(B, C, -1, W)  # (b, c, n, w)
        x_wn_mean = x.mean(dim=-1, keepdim=True).detach()  # (b, c, n, 1)
        x_wn_std = x.std(dim=-1, keepdim=True).detach()  # (b, c, n, 1)

        if self.wne_use_instance_stats:
            x_mean = x_in_mean
            x_std = x_in_std
        else:
            x_mean = x_wn_mean
            x_std = x_wn_std

        if self.wne_use_in:
            x = (x - x_in_mean) / (x_in_std + 1e-8)  # normalize x with mean and std of each window
        else:
            x = (x - x_wn_mean) / (x_wn_std + 1e-8)

        # shape emb
        x = x.view(-1, 1, L)  # (b*c, 1, l)
        x_embed = self.window_embedding(x).transpose(1, 2).view(B, C, N, -1)  # (b*c, 1, l) -> (b*c, d_, n) -> (b, c, n, d_)
        x_embed = self.window_embed_norm(x_embed)

        # mean/std emb
        if self.use_ms_embed:
            x_mean = x_mean.view(B, C, N)
            x_std = x_std.view(B, C, N)
            if self.mean_std_encode == 'ple':
                mean_embeds = self.multi_bias_encoder(x_mean, 'mean').view(B, C, N, -1)
                std_embeds = self.multi_bias_encoder(x_std, 'std').view(B, C, N, -1)
            else:
                mean_embeds = self.multi_bias_encoder(x_mean).view(B, C, N, -1)
                std_embeds = self.multi_bias_encoder(x_std).view(B, C, N, -1)
            xms_embed = torch.cat((x_embed, mean_embeds, std_embeds), dim=-1)  # (b, c, n, d_+2d')
            x_embed = self.ms_embed_fc(xms_embed)  # (b, c, n, d_+2d') -> (b, c, n, d_)

        # multi channels
        if self.use_channel_emb_sum:
            x_embed = x_embed.sum(dim=1)
            x_embed = self.window_embed_norm(x_embed).transpose(1, 2)
        else:
            x_embed = x_embed.transpose(1, 2).contiguous().view(B, N, -1)  # (b, c, n, d_) -> (b, n, c, d_) -> (b, n, c*d_)
            x_embed = self.channel_embedding(x_embed).transpose(1, 2)  # (b, n, c*d_) -> (b, d, n)

        return x_embed
