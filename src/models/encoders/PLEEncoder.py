import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import torch
from torch import nn
import torch.nn.functional as F


class PLEEncoder(nn.Module):
    """Encode numeric feature to embedding.

    Referred from https://arxiv.org/abs/2203.05556 and https://github.com/Yura52/tabular-dl-num-embeddings.
    """
    def __init__(self, config):
        super(PLEEncoder, self).__init__()

        self.load_bin_edges = config.load_bin_edges
        self.bin_edges_dir = config.bin_edges_dir
        self.output_dir = config.output_dir
        self.save_bin_edges = config.save_bin_edges
        self.bins_mode = config.bins_mode
        self.bins_count = config.bins_count
        self.out_channels = 0
        self.use_win_stats = True if config.encoder == 'wne' else False
        self.window_size = config.window_size
        self.transform_size = config.transform_size

    def get_win_stats(self, x):
        fixed_x = []
        for sample in x:
            sample = sample[:, ~torch.all(sample.isnan(), dim=0)]
            sample = torch.nan_to_num(sample, nan=0.)
            fixed_sample = F.interpolate(sample.unsqueeze(0), size=self.transform_size)
            fixed_x.append(fixed_sample)
        fixed_samples = torch.cat(fixed_x, dim=0)
        B, C, L = fixed_samples.shape
        fixed_samples = fixed_samples.view(B, C, -1, self.window_size)  # (b, c, n, w)
        x_mean = fixed_samples.mean(dim=-1, keepdim=True).squeeze(-1) # (b, c, n)
        x_std = fixed_samples.std(dim=-1, keepdim=True).squeeze(-1)  # (b, c, n)

        return x_mean, x_std

    def load_bins(self):
        try:
            f = os.path.join(self.bin_edges_dir)
            bin_edges = torch.load(f)
            self.mean_bin_edges = bin_edges['mean'] if 'mean' in bin_edges else None
            self.std_bin_edges = bin_edges['std'] if 'std' in bin_edges else None
            self.raw_bin_edges = bin_edges['raw'] if 'raw' in bin_edges else None
        except:
            print(f"Load file {f} bin edges failed!")
            return False
        print(f"Load file {f} bin edges successful!")
        print(self.mean_bin_edges)
        print(self.std_bin_edges)
        return True

    def save_bins(self, bin_edges):
        f = os.path.join(self.output_dir, 'bin_edges.pth')
        if os.path.exists(f):
            os.remove(f)
        torch.save(bin_edges, f)

    def get_bin_edges(self, samples, targets):
        if self.load_bin_edges:
            load_success = self.load_bins()
            if load_success:
                return
        
        samples_dim = samples.ndim
        if samples_dim < 3:
            raise ValueError(f'Dim of `samples` must not be less than 3, got {samples_dim}!')
        
        bin_dict = {}
        bin_edges = {}
        if self.use_win_stats:
            x_mean, x_std = self.get_win_stats(samples)
            bin_dict['mean'] = x_mean
            bin_dict['std'] = x_std
            bin_edges['mean'] = []
            bin_edges['std'] = []
        else:
            bin_dict['raw'] = samples
            bin_edges['raw'] = []

        for feature_name, feature_values in bin_dict.items():
            feature_values = feature_values.transpose(1, 2).numpy()
            B, L, C = feature_values.shape
            for feature_idx in range(C):
                train_column = feature_values[:, :, feature_idx].copy()
                if self.bins_mode == 'target':
                    tree = (
                        DecisionTreeClassifier(max_leaf_nodes=self.bins_count // 2)   # tune trees' params
                        .fit(train_column.reshape(B, L), targets)  # use a series of selected feature to predict label
                        .tree_
                    )
                    tree_thresholds = []
                    for node_id in range(tree.node_count):
                        # if tree.children_left[node_id] != tree.children_right[node_id]:
                        tree_thresholds.append(tree.threshold[node_id])
                    tree_thresholds.append(train_column.min())
                    tree_thresholds.append(train_column.max())
                    bin_edges[feature_name].append(np.array(sorted(tree_thresholds)))
                else:   # 'quantile'
                    quantiles = np.linspace(0.0, 1.0, self.bins_count + 1)
                    bin_edges[feature_name].append(np.quantile(train_column, quantiles))
                self.out_channels += len(bin_edges[feature_name][-1]) - 1
        self.mean_bin_edges = bin_edges['mean'] if 'mean' in bin_edges else None
        self.std_bin_edges = bin_edges['std'] if 'std' in bin_edges else None
        self.raw_bin_edges = bin_edges['raw'] if 'raw' in bin_edges else None
        if self.save_bin_edges:
            self.save_bins(bin_edges)

    def forward(self, samples: torch.Tensor, name='raw'):
        m = {
            'raw': self.raw_bin_edges,
            'mean': self.mean_bin_edges,
            'std': self.std_bin_edges
        }
        bin_edges = m[name]
        
        B, C, L = samples.shape
        assert C == len(bin_edges), f'samples feature dim {C} is not equal to bins feature dim {len(bin_edges)}!'
        _bins = []
        _bin_values = []
        samples_c = samples.transpose(1, 2).view(-1, C).cpu() # (B * L, C)
        for feature_idx in range(len(bin_edges)):
            # samples feature to bins
            _bins.append(
                np.digitize(
                    samples_c[:, feature_idx],
                    np.r_[-np.inf, bin_edges[feature_idx][1:-1], np.inf]
                ) - 1
            )
            # get bin sizes
            feature_bin_sizes = (
                bin_edges[feature_idx][1:] - bin_edges[feature_idx][:-1]
            )
            # get bin values
            _bin_values.append(
                (
                    samples_c[:, feature_idx]
                    - bin_edges[feature_idx][_bins[:][feature_idx]]
                )
                / feature_bin_sizes[_bins[:][feature_idx]]
            )
        n_bins = max(map(len, bin_edges)) - 1
        bins = torch.as_tensor(np.stack(_bins, axis=1), dtype=torch.int64)
        del _bins
        bin_values = torch.as_tensor(np.stack(_bin_values, axis=1), dtype=torch.float32)
        del _bin_values
        bin_mask_ = torch.eye(n_bins)[bins]   # (B * L, C, n_bins)
        x = bin_mask_ * bin_values.unsqueeze(dim=-1)    # fill bin_values in hit bins
        previous_bins_mask = torch.arange(n_bins).expand(1, 1, n_bins).repeat(
            B * L, C, 1
        ) < bins.reshape(B * L, C, 1)
        x[previous_bins_mask] = 1.0 # fill ones in left bins
        x = x.view(B, L, -1).transpose(1, 2).contiguous().to(samples.device)

        return x
