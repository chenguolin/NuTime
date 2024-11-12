from src.models.networks import *
from src.models.ssl import *


def get_model(config):
    # set out_dim for classification task
    if config.task == 'cls':
        config.out_dim = config.num_classes

    if config.model == 'mlp-baseline':
        model = MLPBaseline(
            in_channels=config.model_series_size*config.num_channels,
            out_channels=config.out_dim, hidden_size=config.hidden_size
        )
    elif config.model == 'fcn-baseline':
        model = FCNBaseline(
            in_channels=config.num_channels, out_channels=config.out_dim
        )
    elif config.model == 'resnet-baseline':
        model = ResNetBaseline(
            in_channels=config.num_channels, out_channels=config.out_dim
        )
    elif config.model == 'inceptiontime':
        model = InceptionTime(
            in_channels=config.num_channels, out_channels=config.out_dim, norm=config.inception_norm
        )
    elif config.model == 'tcn':
        model = TCN(
            in_channels=config.num_channels, out_channels=config.out_dim, dropout=config.dropout
        )
    elif config.model == 'wint':
        model = WinT(config, 
            series_size=config.model_series_size, in_channels=config.num_channels, window_emb_dim=config.window_emb_dim,
            out_channels=config.out_dim, window_size=config.window_size, stride=config.stride,
            depth=config.transformer_depth, heads=config.transformer_heads, head_dim=config.transformer_head_dim,
            mlp_dim=config.transformer_mlp_dim, pool_mode=config.pool_mode, pe_mode=config.pe_mode,
            max_num_tokens=config.transformer_max_tokens, dropout=config.dropout, pos_dropout=config.pos_dropout,
            attn_dropout=config.attn_dropout, path_dropout=config.path_dropout, init_values=None
        )
    elif config.model == 'gru':
        model = GRUBaseline(
            in_channels=config.num_channels, out_channels=config.out_dim,
            hidden_size=config.hidden_size, layer_num=2
        )
    else:
        raise NotImplementedError(f'Model `{config.model}` is not found!')

    return model


def get_ssl_model(config, backbone):
    if config.ssl_method == 'simclr':
        model = SimCLRModel(backbone=backbone, num_layers=2, use_bn=True, non_linear=True, tau=config.tau)
    elif config.ssl_method == 'moco-v1':
        model = MoCoModel(backbon=backbone, queue_size=65536, momentum=config.moco_momentum, tau=config.tau, projector=False)
    elif config.ssl_method == 'moco-v2':
        model = MoCoModel(backbone=backbone, queue_size=65536, momentum=config.moco_momentum, tau=config.tau, projector=True)
    elif config.ssl_method == 'moco-v3':
        model = MoCoV3Model(backbone=backbone, tau=config.tau)
    elif config.ssl_method == 'byol':
        model = BYOLModel(backbone=backbone)
    elif config.ssl_method == 'simsiam':
        model = SimSiamModel(backbone=backbone)
    else:
        NotImplementedError(f'Self-Supervised Learning method `{config.ssl_method}` is not found!')

    return model
