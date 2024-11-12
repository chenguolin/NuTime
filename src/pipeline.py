import os
import warnings

import torch

from src.config import get_config_from_file
from src.utils.util import *
import src.experiments as experiments

import warnings
warnings.filterwarnings('ignore')


class Pipeline(object):
    def __init__(self, config) -> None:
        self.config = config
        self.logger = get_logger(config)
        self.set_seed()
        self.prepare_env()

    def set_seed(self):
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed_all(self.config.seed)
        if self.config.cuda_determine:  # slower, but more reproducible
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True

    def prepare_env(self):
        # prepare dir
        os.makedirs(self.config.output_dir, exist_ok=True)
        self.config.output_dir = os.path.join(self.config.output_dir, self.config.tag)
        os.makedirs(self.config.output_dir, exist_ok=True)
        # prepare cuda
        ngpus_per_node = torch.cuda.device_count()
        if not torch.cuda.is_available():
            self.logger.info('No GPU available! Using CPU, this will be slow...')
        elif self.config.device is not None:
            self.device = torch.device(self.config.device)
            self.logger.info(f'Using GPU:{self.config.device} for undistributed computation')
            torch.cuda.set_device(self.config.device)
        else:
            self.logger.info(f'Using {ngpus_per_node} GPU(s) for undistributed computation')

    def run_exp(self):
        exp_dict = {
            "cls": "ClassificationExp",
            "ssl": "SelfSupervisedLearningExp",
        }
        self.exp_cls = getattr(experiments, exp_dict[self.config.task])
        exp = self.exp_cls(config=self.config, logger=self.logger)

        acc_results, mf1_results = 0.0, 0.0
        if not self.config.few_shot_learning:
            acc_results, mf1_results = exp.run()
        else:
            # In few shot setting, run several n-way-k-shot tasks.
            task_num = self.config.few_shot_task_num
            acc_results_list, mf1_results_list = [], []
            self.config.patience = self.config.few_shot_patience
            for s in range(task_num):
                self.logger.info(f'Start few shot experiment with task {s+1}/{task_num}')
                self.config.seed = s
                self.set_seed()
                best_test_acc, best_test_mf1 = exp.run()
                acc_results_list.append(best_test_acc)
                mf1_results_list.append(best_test_mf1)
            acc_results = sum(acc_results_list) / task_num
            mf1_results = sum(mf1_results_list) / task_num
            self.logger.info(
                'Few shot training finished! '
                f'Average Acc | Macro-F1 on {task_num} task: {acc_results:.3f}% | {mf1_results:.3f}%')

        return acc_results, mf1_results


def save_success_file(config):
    save_path = f"{config.exp_log_path}/success"
    with open(save_path, 'w') as f:
        f.write("Success")


if __name__ == '__main__':
    config = get_config_from_file()
    pipeline = Pipeline(config)
    save_config(config)
    pipeline.run_exp()
    # save_success_file(config)
