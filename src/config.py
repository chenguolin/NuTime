import os
import json
import argparse


class Config(object):
    def __init__(self):
        # ------- Basic Arguments -------
        self.seed = 0  # random seed
        self.device = None
        self.log_dir = './out'
        self.log_file = self.log_dir + '/log.txt'
        self.save_checkpoint = True
        self.save_preds = False
        self.ssl_save_checkpoint_freq = 10
        self.few_shot_learning = False
        self.few_shot_task_num = 100
        self.trainset_shot_num = 5
        self.testset_shot_num = 5
        self.few_shot_patience = 16

        # ------- Optimization Arguments -------
        self.max_epochs = 500
        self.patience = 16
        self.batch_size = 1024
        self.eval_batch_size = 5120
        self.learning_rate = 2e-3
        self.weight_decay = 2e-4
        self.use_eval = True
        self.use_weighted_sampler = False
        self.transformer_use_lrd = False
        self.window_mask_generator = 'block'
        self.transformer_mask_type = 'none' # or: 'learnable', 'drop'
        self.transformer_mask_scale = 0
        self.transform_size_mode = 'fixed'

        # ------- Ablation Arguments -------
        self.wne_use_in = False
        self.wne_use_instance_stats = False

    def update_by_dict(self, config_dict):
        for key, val in config_dict.items():
            setattr(self, key, val)

    def set_tag(self):
        self.tag = f'{self.task}_{self.dataset}_{self.transform_size}_{self.norm}_{self.encoder}_{self.model}_{self.num_epochs}_{self.seed}'
        if self.model == 'wint':
            if self.scale_weighted_sum:
                self.tag += '_ws'
            else:
                self.tag += '_nws'
            self.tag += f'_d{self.transformer_depth}_h{self.transformer_heads}_hd{self.transformer_head_dim}'
        if self.pretrained_model:
            self.tag += f'_pretrain_{self.load_checkpoint_path.split(os.sep)[-2]}'
            if self.freeze_backbone:
                self.tag += f'_freeze'
            else:
                self.tag += f'_full'

    def to_dict(self):
        return dict(self.__dict__)


def strtobool(str_val):
    """Convert a string representation of truth to true (1) or false (0).
    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    str_val = str_val.lower()
    if str_val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    elif str_val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    else:
        raise ValueError("invalid truth value %r" % (str_val,))


def add_config_to_argparse(config, arg_parser):
    """The helper for adding configuration attributes to the argument parser"""
    for key, val in config.to_dict().items():
        if isinstance(val, bool):
            arg_parser.add_argument('--' + key, type=strtobool, default=val)
        elif isinstance(val, (int, float, str)):
            arg_parser.add_argument('--' + key, type=type(val), default=val)
        else:
            raise Exception('Do not support value ({}) type ({})'.format(val, type(val)))


def get_config_from_command():
    # add arguments to parser
    config = Config()
    parser = argparse.ArgumentParser(description='Time Series Representation')
    # add_config_to_argparse(config, parser)

    # parse arguments from command line
    args = parser.parse_args()
    config.update_by_dict(args.__dict__)
    config.set_tag()

    return config


def get_config_from_file():
    config = Config()
    parser = argparse.ArgumentParser(description='Time Series Representation')

    # test_config
    parser.add_argument('--config_file', type=str, required=False, default='./configs/train_ssl.json')
    args = parser.parse_args()

    config.update_by_dict(
        json.load(open(args.config_file))
    )
    config.set_tag()

    return config
