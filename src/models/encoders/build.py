from torch import nn

from .WindowNormEncoder import WindowNormEncoder
from .PLEEncoder import PLEEncoder
from .PeriodicEncoder import PeriodicEncoder
from .MultiBiasEncoder import MultiBiasEncoderWrapper


def get_encoder(config, dataset):
    out_channels = config.num_channels
    seq_len = config.model_series_size
    if config.encoder == 'identity':
        encoder = nn.Identity()
    elif config.encoder == 'mbe':
        encoder = MultiBiasEncoderWrapper(config)
        out_channels = config.mb_emb_dim
    elif config.encoder == 'wne':
        encoder = WindowNormEncoder(config, dataset.samples, dataset.targets)
        out_channels = config.window_emb_dim
        seq_len = encoder.window_num
    elif config.encoder == 'ple':
        encoder = PLEEncoder(config)
        encoder.get_bin_edges(dataset.samples, dataset.targets)
        out_channels = encoder.out_channels
    elif config.encoder == 'periodic':
        encoder = PeriodicEncoder(config)
        out_channels = encoder.out_channels
    else:
        raise NotImplementedError(f'Model `{config.encoder}` is not found!')

    # resize feature channels
    config.num_channels = out_channels
    config.model_series_size = seq_len

    return encoder
