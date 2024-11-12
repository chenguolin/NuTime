import os
import warnings

from src.config import *
from src.pipeline import *

import warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'


def run(config):
    pipeline = Pipeline(config)
    save_config(config)
    best_test_acc, best_test_mf1 = pipeline.run_exp()
    return best_test_acc, best_test_mf1


TASK = 'ft'  # 'ssl', 'cls', or 'ft'


if __name__ == '__main__':
    # read the default model config
    default_config = json.load(open(f"configs/train_{TASK}.json"))

    config = Config()
    config.update_by_dict(default_config)
    config.set_tag()  # auto set the tag based on the config
    config.transformer_mlp_dim = config.transformer_heads * config.transformer_head_dim

    print(f"Start run {config.dataset}, log: {config.log_file}")
    best_test_acc, best_test_mf1 = run(config)
