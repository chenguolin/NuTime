import functools
import random
import math

from scipy.interpolate import CubicSpline
import numpy as np
import torch
import torch.nn.functional as F


class TSIdentity:
    """Identity transform."""
    def __call__(self, series: torch.Tensor):
        if series.dim() != 2:
            raise ValueError(f'Parameter `series` must be a 2D tensor, but got {series.shape}!')

        return series


class TSNormalize:
    """Normalize a time series to zero mean and unit variance."""
    def __init__(self, mode='instance', eps=1e-8):
        """
        Parameters:
            mode: mode of normalization, [`'instance'`, `'minmax'`, `'global'`, `'identity'`]
            eps (float): epsilon for normalization
        """
        assert mode in ['siminst', 'instance', 'global', 'identity'], f'Parameter `mode` must be one of [`siminst`, `instance`, `global`, `identity`], but got {mode}!'
        self.mode, self.eps = mode, eps

    def __call__(self, series: torch.Tensor):
        if series.dim() != 2:
            raise ValueError(f'Parameter `series` must be a 2D tensor, but got {series.shape}!')

        if self.mode == 'siminst':
            return (series - series.mean(dim=-1, keepdim=True)) / (series.std(dim=-1, keepdim=True) + self.eps)
        else:  # self.mode == 'identity' or 'global' (normalized in the Dataset class)
            return series


class TSRandomVerticalFlip:
    """Randomly flip a time series vertically."""
    def __init__(self, p=0.5):
        assert 0 <= p <= 1, f'Parameter `p` must be in [0, 1], but got {p}!'

        self.p = p

    def __call__(self, series: torch.Tensor):
        if series.dim() != 2:
            raise ValueError(f'Parameter `series` must be a 2D tensor, but got {series.shape}!')

        return -series if torch.rand(1).item() < self.p else series


class TSRandomHorizontalFlip:
    """Randomly flip a time series horizontally."""
    def __init__(self, p=0.5):
        assert 0 <= p <= 1, f'Parameter `p` must be in [0, 1], but got {p}!'

        self.p = p

    def __call__(self, series: torch.Tensor):
        if series.dim() != 2:
            raise ValueError(f'Parameter `series` must be a 2D tensor, but got {series.shape}!')

        return torch.flip(series, dims=[-1]) if torch.rand(1).item() < self.p else series


class TSNoisePad:
    """Pad a time series with random noise or truncate it."""
    def __init__(self, size: int, amplitude=0.01):
        """
        Parameters:
            size (int): expected output size
            amplitude (float): noise amplitude
        """
        self.size = size
        self.amplitude = amplitude

    def __call__(self, series: torch.Tensor):
        if series.dim() != 2:
            raise ValueError(f'Parameter `series` must be a 2D tensor, but got {series.shape}!')

        c, t = series.shape
        if t > self.size:  # truncate
            return series[:, :self.size]
        else:  # pad
            noise = torch.randn(c, self.size - t) * self.amplitude
            return torch.cat((series, noise), dim=1)


class TSAddNoise:
    """Jitter a time series by adding i.i.d. random noises."""
    def __init__(self, mean=0., std_ratio=0.2, point_wise=False):
        """
        Parameters:
            mean (float): mean of the noise
            std (float): standard deviation of the noise
            point_wise (bool): whether to add noise point-wise or not
        """
        self.mean, self.std_ratio = mean, std_ratio
        self.point_wise = point_wise

    def __call__(self, series: torch.Tensor):
        if series.dim() != 2:
            raise ValueError(f'Parameter `series` must be a 2D tensor, but got {series.shape}!')

        std = self.std_ratio * series.std() + 1e-8  # add a small epsilon to avoid zero std

        return series + torch.normal(mean=self.mean, std=std, size=series.shape if self.point_wise else (series.shape[0], 1))


class TSMulNoise:
    """Scale a time series by multiplying random noises."""
    def __init__(self, mean=1., std=0.2, point_wise=False):
        """
        Parameters:
            mean (float): mean of the noise
            std (float): standard deviation of the noise
            point_wise (bool): whether to apply the noise to each point
        """
        self.mean, self.std = mean, std
        self.point_wise = point_wise

    def __call__(self, series: torch.Tensor):
        if series.dim() != 2:
            raise ValueError(f'Parameter `series` must be a 2D tensor, but got {series.shape}!')

        return series * torch.normal(mean=self.mean, std=self.std, size=series.shape if self.point_wise else (series.shape[0], 1))


class TSCenterCrop:
    """Crop the central portion of a time series."""
    def __init__(self, size: int):
        """
        Parameters:
            size (int): expected output size
        """
        self.size = size

    def __call__(self, series: torch.Tensor):
        if series.dim() != 2:
            raise ValueError(f'Parameter `series` must be a 2D tensor, but got {series.shape}!')

        length = series.shape[-1]
        start_index = round((length - self.size) / 2)

        return series[..., start_index : start_index+self.size]


class TSRandomResizedCrop:
    """Crop a random portion of a time series and resize it to the given size."""
    def __init__(self, size: int, scale=[0.8, 1.], mode='linear'):
        """
        Parameters:
            size (int): expected output size
            scale (list | tuple): specifies the lower and upper bounds for the random sequence of the crop, before resizing;
                the scale is defined with respect to the length of the original time series
            mode (str): desired interpolation mode for resizing [`'linear'`, `'nearest'`, `'area'`]
        """
        assert len(scale) == 2, f'Parameter `scale` should be a list/tuple of length 2, but got {len(scale)}!'
        if (scale[0] > scale[1]) or (scale[0] < 0) or (scale[1] > 1):
            raise ValueError(f'Parameter `scale` must be a list/tuple of two numbers in the range [0, 1] and in non-decreasing order, but got {scale}!')
        assert mode in ['linear', 'nearest', 'area'], f'Parameter `mode` must be one of [`linear`, `nearest`, `area`], but got {mode}!'

        self.size = size
        self.scale = scale
        self.mode = mode

    def __call__(self, series: torch.Tensor):
        if series.dim() != 2:
            raise ValueError(f'Parameter `series` must be a 2D tensor, but got {series.shape}!')

        length = series.shape[-1]
        target_length = round(length * torch.empty(1).uniform_(self.scale[0], self.scale[1]).item())
        start_index = torch.randint(0, length-target_length+1, (1,)).item()
        cropped_series = series[..., start_index : start_index+target_length]
        # expected inputs of F.interpolate() are 3-D, 4-D or 5-D in shape
        resized_series = F.interpolate(cropped_series.unsqueeze(0), size=self.size, mode=self.mode, align_corners=None if self.mode in ['nearest', 'area'] else False).squeeze(0)

        return resized_series


class TSRandomCutout:
    """Cut out a random portion of a time series."""
    def __init__(self, scale=[0., 0.2], fill_mode='normal'):
        """
        Parameters:
            scale (list | tuple): specifies the lower and upper bounds for the random sequence of the cutout
        """
        assert len(scale) == 2, f'Parameter `scale` should be a list/tuple of length 2, but got {len(scale)}!'
        if (scale[0] > scale[1]) or (scale[0] < 0) or (scale[1] > 1):
            raise ValueError(f'Parameter `scale` must be a list/tuple of two numbers in the range [0, 1] and in non-decreasing order, but got {scale}!')
        assert fill_mode in ['normal', 'unit_normal', 'zero'], f'Parameter `fill_mode` must be one of [`normal`, `unit_normal`, `zero`], but got {fill_mode}!'

        self.scale = scale
        self.fill_mode = fill_mode

    def __call__(self, series: torch.Tensor):
        if series.dim() != 2:
            raise ValueError(f'Parameter `series` must be a 2D tensor, but got {series.shape}!')

        length = series.shape[-1]
        target_length = round(length * torch.empty(1).uniform_(self.scale[0], self.scale[1]).item())
        start_index = torch.randint(0, length-target_length+1, (1,)).item()

        fill_values = torch.randn(series.shape[0], target_length)
        if self.fill_mode == 'normal':
            fill_values = fill_values * series.std(dim=-1)[:, None, None] + series.mean(dim=-1)[:, None, None]
        elif self.fill_mode == 'unit_normal':
            pass  # fill_values already has unit normal distribution
        else:  # self.fill_mode == 'zero'
            fill_values.fill_(0.)

        series[..., start_index : start_index+target_length] = fill_values

        return series


class TSRandomMask:
    """Randomly mask a time series."""
    def __init__(self, scale=[0., 0.2], mask_mode='sequence', win_size=16, window_mask_generator='block'):
        assert len(scale) == 2, f'Parameter `scale` should be a list/tuple of length 2, but got {len(scale)}!'
        if (scale[0] > scale[1]) or (scale[0] < 0) or (scale[1] > 1):
            raise ValueError(f'Parameter `scale` must be a list/tuple of two numbers in the range [0, 1] and in non-decreasing order, but got {scale}!')
        assert mask_mode in ['window', 'sequence'], f'Parameter `mask_mode` must be one of [`window`, `sequence`], but got {mask_mode}!'
        assert mask_mode != 'window' or win_size > 0, f'Parameter `win_size` must be a positive integer if `mask_mode` is `window`, but got {win_size}!'

        self.scale = scale
        self.mask_mode = mask_mode
        self.win_size = win_size
        self.window_mask_generator = window_mask_generator

    def get_random_mask(self, series, num_wins, mask_wins):
        noise = torch.rand(series.shape[0], num_wins)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        mask = torch.zeros((series.shape[0], num_wins))
        mask[:, :mask_wins] = 1.
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return mask

    def get_block_mask(self, series, num_wins, mask_wins):
        """1D block-wise masking for time series.
        Referred from https://github.com/microsoft/unilm/blob/master/beit/masking_generator.py.
        """
        mask = torch.zeros((num_wins))
        mask_count = 0
        while mask_count < mask_wins:
            delta = 0
            for attempt in range(10):
                max_mask_wins = mask_wins - mask_count
                block_len = random.uniform(0, max_mask_wins)
                aspect_ratio = random.uniform(0.5, 2)
                l = int(round(math.sqrt(block_len * aspect_ratio)))
                if l < num_wins:
                    left = random.randint(0, num_wins - l)
                    num_masked = mask[left: left + l].sum()
                    if 0 < l - num_masked <= max_mask_wins:
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
        mask = mask.repeat(series.shape[0], 1)

        return mask

    def __call__(self, series: torch.Tensor):
        if series.dim() != 2:
            raise ValueError(f'Parameter `series` must be a 2D tensor, but got {series.shape}!')

        length = series.shape[-1]
        target_length = round(length * torch.empty(1).uniform_(self.scale[0], self.scale[1]).item())
        if target_length <= 1: return series  # no masking needed

        if self.mask_mode == 'window':
            assert length % self.win_size == 0, f'Length of time series must be divisible by `win_size` for `mask_mode` == `window`, but got {length}!'
            series_wins = series.view(-1, length // self.win_size, self.win_size)
            num_wins, mask_wins = length // self.win_size, target_length // self.win_size

            if self.window_mask_generator == 'random':
                mask = self.get_random_mask(series, num_wins, mask_wins)
            elif self.window_mask_generator == 'block':
                mask = self.get_block_mask(series, num_wins, mask_wins)

            mask = mask.unsqueeze(-1)
            fill_values = torch.randn(series.shape[0], num_wins, self.win_size)
            fill_values = fill_values * series_wins.std(dim=-1, keepdim=True) + series_wins.mean(dim=-1, keepdim=True)
            series_wins = series_wins * (1. - mask) + fill_values * mask
            series = series_wins.view_as(series)
        else:  # self.mask_mode == 'sequence'
            start_index = torch.randint(0, length-target_length+1, (1,)).item()
            seq_mean = series[..., start_index : start_index+target_length].mean(dim=-1, keepdim=True)
            seq_std = series[..., start_index : start_index+target_length].std(dim=-1, keepdim=True)

            fill_values = torch.randn(series.shape[0], target_length)
            fill_values = fill_values * seq_std + seq_mean
            series[..., start_index : start_index+target_length] = fill_values

        return series


class TSDropout:
    """Dropout a time series."""
    def __init__(self, p=0.1):
        """
        Parameters:
            p (float): dropout probability
        """
        assert 0 <= p <= 1, f'Parameter `p` must be a number in the range [0, 1], but got {p}!'

        self.p = p

    def __call__(self, series: torch.Tensor):
        if series.dim() != 2:
            raise ValueError(f'Parameter `series` must be a 2D tensor, but got {series.shape}!')

        return F.dropout(series, p=self.p, training=True)


class TSFreqDropout:
    """Dropout the amplitude and phase spectrum of a time series after Fourier Transformation."""
    def __init__(self, amp_dropout=0.1, pha_dropout=0.):
        """
        Parameters:
            amp_dropout (float): probability of dropping the amplitude spectrum
            pha_dropout (float): probability of dropping the phase spectrum
        """
        assert 0 <= amp_dropout <= 1, f'Parameter `amp_dropout` must be a number in the range [0, 1], but got {amp_dropout}!'
        assert 0 <= pha_dropout <= 1, f'Parameter `pha_dropout` must be a number in the range [0, 1], but got {pha_dropout}!'

        self.amp_dropout = amp_dropout
        self.pha_dropout = pha_dropout

    def __call__(self, series: torch.Tensor):
        if series.dim() != 2:
            raise ValueError(f'Parameter `series` must be a 2D tensor, but got {series.shape}!')

        # compute the amplitude and phase spectrum of the time series after Fourier Transformation
        freq_series = torch.fft.rfft(series)
        amplitudes, phases = freq_series.abs(), freq_series.angle()
        R_signals, I_signals = freq_series.real // freq_series.real.abs(), freq_series.imag // freq_series.imag.abs()

        amplitudes = F.dropout(amplitudes, p=self.amp_dropout, training=True) * (1 - self.amp_dropout)
        phases = F.dropout(phases, p=self.pha_dropout, training=True) * (1 - self.pha_dropout)

        # reconstruct the time series from the pertrubed amplitude and phase spectrum
        reconstructed_R = ((1 / (torch.tan(phases)**2 + 1)) * amplitudes**2)**0.5 * R_signals
        reconstructed_I = ((torch.tan(phases)**2 / (torch.tan(phases)**2 + 1)) * amplitudes**2)**0.5 * I_signals
        reconstructed_R.nan_to_num_()  # replace NaN with 0
        reconstructed_I.nan_to_num_()  # replace NaN with 0
        reconstructed_freq_series = reconstructed_R + 1j * reconstructed_I

        return torch.fft.irfft(reconstructed_freq_series)


class TSMagWarp:
    """Warp the magnitude of a time series by multiplying by a curve created by cubic spline interpolation.
    The curve can be considered as smoothly-varing noises.
    """
    def __init__(self, std=0.2, n_knots=4):
        """
        Parameters:
            std (float): standard deviation of the noise
            n_knots (int): number of knots to use in cubic spline interpolation
        """
        self.std = std
        self.n_knots = n_knots

    def __call__(self, series: torch.Tensor):
        if series.dim() != 2:
            raise ValueError(f'Parameter `series` must be a 2D tensor, but got {series.shape}!')

        warp_step = np.linspace(start=0, stop=series.shape[-1]-1, num=self.n_knots+2)
        random_warp = np.random.normal(loc=1., scale=self.std, size=self.n_knots+2)
        f = CubicSpline(x=warp_step, y=random_warp)
        warper = f(np.arange(series.shape[-1]))

        return series * torch.from_numpy(warper).float()


class TSTimeWarp:
    """Warp a time series on the dimension of time."""
    def __init__(self, std=0.2, n_knots=4):
        """
        Parameters:
            std (float): standard deviation of the noise
            n_knots (int): number of knots to use in cubic spline interpolation
        """
        self.std = std
        self.n_knots = n_knots

    def __call__(self, series: torch.Tensor):
        if series.dim() != 2:
            raise ValueError(f'Parameter `series` must be a 2D tensor, but got {series.shape}!')

        warp_step = np.linspace(start=0, stop=series.shape[-1]-1, num=self.n_knots+2)
        random_warp = np.random.normal(loc=1., scale=self.std, size=self.n_knots+2)
        f = CubicSpline(x=warp_step, y=random_warp)
        warper = f(np.arange(series.shape[-1]))

        warper_cum = np.cumsum(warper)
        # normalize the cumulative sum to be in the range [0, 1]
        warper_cum -= warper_cum[0]
        warper_cum /= warper_cum[-1]

        warper_time = warper_cum * (series.shape[-1] - 1)
        f_time = CubicSpline(np.arange(series.shape[-1]), series.numpy(), axis=-1)  # interpolation based on the original time series

        return torch.from_numpy(f_time(warper_time)).float()


class TSWinWarp:
    """Warp a slice of time series on the dimension of time."""
    def __init__(self, ratio=0.5, scale=[0.5, 2.]):
        """
        Parameters:
            ratio (float): ratio of the window size to the time series length
            scale (list | tuple): range of the random warp
        """
        assert len(scale) == 2, f'Parameter `scale` should be a list/tuple of length 2, but got {len(scale)}!'
        if (scale[0] > scale[1]) or (scale[0] < 0) or (scale[1] < 0):
            raise ValueError(f'Parameter `scale` must be a list/tuple of two non-negative numbers in non-decreasing order, but got {scale}!')

        self.ratio = ratio
        self.scale = scale

    def __call__(self, series: torch.Tensor):
        if series.dim() != 2:
            raise ValueError(f'Parameter `series` must be a 2D tensor, but got {series.shape}!')

        win_len = int(series.shape[-1] * self.ratio + 0.5)
        start_index = np.random.randint(low=0, high=series.shape[-1]-win_len+1)
        # warp a random slice of time series
        warper = np.ones(series.shape[-1])
        warper[start_index:start_index+win_len] = np.random.uniform(low=self.scale[0], high=self.scale[1], size=1)

        warper_cum = np.cumsum(warper)
        # normalize the cumulative sum to be in the range [0, 1]
        warper_cum -= warper_cum[0]
        warper_cum /= warper_cum[-1]

        warper_time = warper_cum * (series.shape[-1] - 1)
        f_time = CubicSpline(np.arange(series.shape[-1]), series.numpy(), axis=-1)  # interpolation based on the original time series

        return torch.from_numpy(f_time(warper_time)).float()


# new transformations based on defined ones
TSZeroPad = functools.partial(TSNoisePad, amplitude=0.)
TSResize = functools.partial(TSRandomResizedCrop, scale=[1., 1.])


class TSCompose:
    """Compose several time series transforms together."""
    def __init__(self, transforms: list):
        """
        Parameters:
            transforms (list): list of transforms to compose
        """
        self.transforms = transforms

    def __call__(self, series: torch.Tensor):
        for t in self.transforms:
            series = t(series)

        return series


class TSRandomApply:
    """Apply some transformations randomly."""
    def __init__(self, transforms: list, p: float):
        """
        Parameters:
            transforms (list): list of transforms to apply randomly
            p (float): probability of applying these transforms
        """
        self.transforms = transforms
        self.p = p

    def __call__(self, series: torch.Tensor):
        if torch.rand(1).item() < self.p:
            for t in self.transforms:
                series = t(series)

        return series


class TSRandomOrder:
    """Apply a list of transformations in a random order."""
    def __init__(self, transforms: list):
        """
        Parameters:
            transforms (list): list of transforms to apply in a random order
        """
        self.transforms = transforms
        self.order = np.random.permutation(len(transforms))

    def __call__(self, series: torch.Tensor):
        for i in self.order:
            series = self.transforms[i](series)

        return series


class TSRandomChoice:
    """Randomly choose one transformation from a list."""
    def __init__(self, transforms: list, p=None):
        """
        Parameters:
            transforms (list): list of transforms to choose from
            p (list | None): probability of choosing each transformation
        """
        self.transforms = transforms
        if p is not None:
            assert isinstance(p, list), f'Parameter `p` must be a list, but got {type(p)}!'
            assert len(p) == len(transforms), f'Length of `p` must be equal to the length of `transforms`, but got {len(p)} and {len(transforms)}!'
            assert np.all(np.array(p) >= 0) and np.all(np.array(p) <= 1), f'Parameter `p` must be a list of non-negative numbers in [0, 1], but got {p}!'
            assert np.sum(p) == 1, f'Sum of `p` must be 1, but got {np.sum(p)}!'
        self.p = p

    def __call__(self, series: torch.Tensor):
        t = np.random.choice(self.transforms, p=self.p)

        return t(series)


class TSTwoViewsTransform:
    """Transform a time series into two views."""
    def __init__(self, base_transform: TSCompose):
        self.base_transform = base_transform

    def __call__(self, series: torch.Tensor):
        return [self.base_transform(series), self.base_transform(series)]
