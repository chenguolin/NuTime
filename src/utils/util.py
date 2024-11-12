import os
import sys
import random
import json
import logging

import numpy as np
from sklearn.metrics import classification_report
import torch
import torch.distributed as dist
from torch import optim, nn

from timm.optim import Lars, Lamb
from timm.scheduler import StepLRScheduler, PlateauLRScheduler, CosineLRScheduler
from timm.loss import LabelSmoothingCrossEntropy


class EarlyStopping:
    def __init__(self, patience=16, delta=0, best_score=None):
        self.patience = patience
        self.counter = 0
        self.best_score = best_score
        self.early_stop = False
        self.score_max = -np.Inf
        self.delta = delta

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score <= self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def get_logger(config):
    """Get a message logger.

    Parameters:
        config (argparse.Namespace): arguments

    Returns:
        logging.Logger: a message logger
    """
    logger = logging.getLogger(config.tag)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # do not print log in the root logger

    fmt = '[%(asctime)s] (%(name)s): %(message)s'
    # fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    if config.color:
        from termcolor import colored
        color_fmt = colored('[%(asctime)s]', 'green') + colored('(%(name)s)', 'yellow') + ': %(message)s'
    else:
        color_fmt = fmt

    # console handler
    # if config.rank == 0 or config.rank == -1:  # master process
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(console_handler)
    # file handlers
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    file_handler = logging.FileHandler(config.log_file, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger


def dump_json_to(obj, fpath, indent=2, ensure_ascii=False, **kwconfig):
    """The helper for dumping json into the given file path"""
    with open(fpath, 'w') as fout:
        json.dump(obj, fout, indent=indent, ensure_ascii=ensure_ascii, **kwconfig)


class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0

    def synchronize_between_processes(self):
        """Warning: does not synchronize the current value (i.e., self.val)!"""
        if not is_dist_avail_and_initialized(): return

        t = torch.tensor([self.sum, self.count], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.sum = t[0]
        self.count = t[1]
        self.avg = self.sum / self.count if self.count > 0 else 0


def is_dist_avail_and_initialized():
    """Check if distributed training is available and initialized."""
    return True if dist.is_available() and dist.is_initialized() else False


@torch.no_grad()
def concat_all_gather(x: torch.Tensor):
    """Performs all_gather operation on the provided tensors. Warning: torch.distributed.all_gather has no gradient."""
    world_size = dist.get_world_size() if is_dist_avail_and_initialized() else 1
    if world_size > 1:
        tensors_gather = [torch.ones_like(x) for _ in range(world_size)]
        dist.all_gather(tensors_gather, x, async_op=False)
        x = torch.cat(tensors_gather, dim=0)

    return x


def set_seed(seed, cuda_determine=False):
    """Set random seed for reproducibility. Refer to https://pytorch.org/docs/stable/notes/randomness.html.

    Parameters:
        seed (int): random seed
        cuda_deteremine (bool): if True, operate deterministic on cuda
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    if cuda_determine:  # slower, but more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, but less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def save_config(config):
    """Save arguments to a file.

    Parameters:
        config (argparse.Namespace): arguments

    Returns:
        str: saved arguments in text format
    """
    save_path = os.path.join(config.output_dir, 'config.txt')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    dump_json_to(vars(config), save_path)
    print("save config path:", save_path)


def save_model_architecture(config, model: torch.nn.Module):
    """Save model architecture to a file.

    Parameters:
        config (argparse.Namespace): arguments
        model (nn.Module): model

    Returns:
        str: saved model architecture in text format
    """
    num_params = sum(p.numel() for p in model.parameters())
    save_path = os.path.join(config.output_dir, 'architecture.txt')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print("save architecture path:", save_path)

    message = str(model) + f'\nNumber of parameters: {num_params}\n'
    with open(save_path, 'w') as f:
        f.write(message)

    return message


def save_result(config, type: str, logits_tensor: torch.Tensor, 
                preds_tensor: torch.Tensor, trues_tensor: torch.Tensor, verbose=True):
    """Save result to a file.

    Parameters:
        config (argparse.Namespace): arguments
        type (str): 'test' or 'val'
        logits_tensor (torch.Tensor): predicted logits
        preds_tensor (torch.Tensor): predicted labels
        trues_tensor (torch.Tensor): true labels
        verbose (bool): if True, print result

    Returns:
        tuple (str, float): saved result in text format and mean accuracy over classes
    """
    result_save_path = os.path.join(config.output_dir, 'result.txt')
    os.makedirs(os.path.dirname(result_save_path), exist_ok=True)

    # save results
    report = classification_report(trues_tensor.cpu().numpy(), preds_tensor.cpu().numpy(), target_names=config.classes, digits=5, output_dict=True)
    if verbose:
        print(classification_report(trues_tensor.cpu().numpy(), preds_tensor.cpu().numpy(), target_names=config.classes, digits=5))
    message = ''
    for k in sorted(config.classes):
        d = report[k]
        message += f'{k:<2}: Precision {d["precision"]*100:>7.3f}% | Recall {d["recall"]*100:>7.3f}% | F1 {d["f1-score"]*100:>7.3f}% | Support {int(d["support"]):>7d}\n'
    message += '\n\n'
    message += f'Accuracy     {report["accuracy"]*100:>7.3f}%\n'
    message += f'Macro Avg    Precision {report["macro avg"]["precision"]*100:>7.3f}% | Recall {report["macro avg"]["recall"]*100:>7.3f}% | F1 {report["macro avg"]["f1-score"]*100:>7.3f}%\n'
    message += f'Weighted Avg Precision {report["weighted avg"]["precision"]*100:>7.3f}% | Recall {report["weighted avg"]["recall"]*100:>7.3f}% | F1 {report["weighted avg"]["f1-score"]*100:>7.3f}%\n'
    with open(result_save_path, 'w') as f:
        f.write(message)

    # save preds
    if config.save_preds and type == 'test':
        obj = {}
        obj["preds"] = logits_tensor
        obj["targets"] = trues_tensor
        output_file = os.path.join(config.output_dir, 'preds.pth')
        if os.path.exists(output_file):
            os.remove(output_file)
        torch.save(obj, output_file)

    return message, report['macro avg']['f1-score']*100


@torch.no_grad()
def accuracy(logits: torch.Tensor, targets: torch.Tensor, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k.

    Parameters:
        logits (torch.Tensor): model outputs (float)
        targets (torch.Tensor): targets (int)
        topk (tuple): top-k predictions

    Returns:
        list: top-k accuracies
    """
    maxk = max(topk)
    batch_size = logits.shape[0]

    _, preds = logits.topk(maxk, dim=1, largest=True, sorted=True)  # default parameters
    preds = preds.t()  # (maxk, batch_size)
    correct = preds.eq(targets.view(1, -1).expand_as(preds))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size).item())

    return res


def sanity_check(state_dict: dict, pretrained_path: str):
    """Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).

    Parameters:
        state_dict (dict): model state dict
        pretrained_path (str): path to load pretrained model checkpoint
    """
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    state_dict_pre = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        # only ignore the fc layer
        if 'fc' in k: continue
        # ignore position embeddings for simplicity, as PE may be interpolated for different input sizes
        if k == 'pos_embed': continue
        # name in pretrained model
        try:
            k_pre = 'backbone.' + k
            assert (state_dict[k].cpu() == state_dict_pre[k_pre]).all(), f'Model parameter `{k}` is changed in linear evaluation!'
        except KeyError:
            assert (state_dict[k].cpu() == state_dict_pre[k]).all(), f'Model parameter `{k}` is changed in linear evaluation!'


def get_parameter_groups(config, model, skip_wd_list=(), verbose=True):
    """Get parameter groups for layer-wise learning rate decay and specified weight decay.

    Referred from https://github.com/microsoft/unilm/blob/master/beit.
    """
    parameter_group_names = {}  # message for debugging
    parameter_group_vars = {}
    assigner = LayerDecayValueAssigner([config.llrd_rate ** (config.transformer_depth+1-i) for i in range(config.transformer_depth+2)])

    for name, param in model.named_parameters():
        if not param.requires_grad:  # frozen parameters
            continue

        if name.startswith("module.0"):  # encoder model need weight decay 
            group_name = 'wd'
            this_weight_decay = config.weight_decay
        elif (param.dim() == 1) or ('bias' in name) or ('norm' in name) or (name in skip_wd_list):
            group_name = 'no_wd'
            this_weight_decay = 0.
        else:
            group_name = 'wd'
            this_weight_decay = config.weight_decay

        block_id = assigner.get_block_id(name)
        group_name = f'layer_{block_id}_{group_name}'

        # spesific learning rate for position embeddings
        if config.pelr is not None and name == 'pos_embed':
            parameter_group_names['pe_no_wd'] = {
                'weight_decay': this_weight_decay,
                'param_names': [name],
                'lr': config.pelr
            }
            parameter_group_vars['pe_no_wd'] = {
                'weight_decay': this_weight_decay,
                'params': [param],
                'lr': config.pelr
            }
            continue

        if group_name not in parameter_group_names:
            scale = assigner.get_scale(block_id)
            parameter_group_names[group_name] = {
                'weight_decay': this_weight_decay,
                'param_names': [],
                'lr': scale * config.lr
            }
            parameter_group_vars[group_name] = {
                'weight_decay': this_weight_decay,
                'params': [],
                'lr': scale * config.lr
            }

        parameter_group_names[group_name]['param_names'].append(name)
        parameter_group_vars[group_name]['params'].append(param)
    if verbose:
        import json
        print(f'Param Groups = {json.dumps(parameter_group_names, indent=2)}')

    return list(parameter_group_vars.values())


class LayerDecayValueAssigner:
    """Assign layer-wise learning rate decay values.

    Referred from https://github.com/microsoft/unilm/blob/master/beit.
    """
    def __init__(self, values: list):
        self.values = values  # decay values for eacg layer

    def get_scale(self, block_id: int):
        return self.values[block_id]

    def get_block_id(self, var_name):
        return get_block_id_for_wint(var_name, len(self.values))


def get_block_id_for_wint(var_name: str, max_num: int):
    """Get block id for WinT model, which is used for layer-wise learning rate decay.

    Referred from https://github.com/microsoft/unilm/blob/master/beit.
    """
    if 'cls_token' in var_name:
        return 0  # smallest learing rate
    elif 'transformer.layers' in var_name: 
        name_list = var_name.split('.')
        layer_idx = name_list.index('transformer') + 2
        block_id = int(name_list[layer_idx]) // 6  # 6 layers per transformer block (attn, layerscale1, droppath1, ff, layerscale2, droppath2)
        return block_id + 1  # start from 1
    else:  # fc layer
        return max_num - 1  # largest learning rate


def get_optimizer(config, model):
    """Return a optimizer.

    Parameters:
        config (argparse.Namespace): arguments
        model (nn.Module): model

    Returns:
        torch.optim.Optimizer: a optimizer
    """
    # get update parameters
    if config.transformer_use_lrd and config.model == 'wint' and config.pretrained_model:  # for Transformer model: layer-wise learning rate decay and specified weight decay
        skip_wd_list = ['pos_embed', 'cls_token']
        # input_model = model.wint if config.model == 'mwint' else model
        update_parameters = get_parameter_groups(config, model, skip_wd_list)
        config.weight_decay = 0.  # no weight decay for other parameters
    else:
        update_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))

    if config.optimizer == 'adam':
        optimizer = optim.Adam(
            update_parameters, lr=config.lr,
            betas=(config.beta1, config.beta2), weight_decay=config.weight_decay
        )
    elif config.optimizer == 'sgd':
        optimizer = optim.SGD(
            update_parameters, lr=config.lr,
            momentum=config.momentum, weight_decay=config.weight_decay, nesterov=config.nesterov
        )
    elif config.optimizer == 'adadelta':
        optimizer = optim.Adadelta(
            update_parameters, lr=config.lr,
            rho=config.momentum
        )
    elif config.optimizer == 'adamw':
        optimizer = optim.AdamW(
            update_parameters, lr=config.lr,
            betas=(config.beta1, config.beta2), weight_decay=config.weight_decay
        )
    elif config.optimizer == 'lars':
        optimizer = Lars(
            update_parameters, lr=config.lr,
            momentum=config.momentum, weight_decay=config.weight_decay, nesterov=config.nesterov
        )
    elif config.optimizer == 'lamb':
        optimizer = Lamb(
            update_parameters, lr=config.lr,
            betas=(config.beta1, config.beta2), weight_decay=config.weight_decay
        )
    else:
        raise NotImplementedError(f'Optimizer `{config.optimizer}` is not found!')

    return optimizer


def get_scheduler(config, optimizer):
    """Return a scheduler.

    Parameters:
        config (argparse.Namespace): arguments
        optimizer (torch.optim.Optimizer): optimizer

    Returns:
        torch.optim.lr_scheduler: a scheduler
    """
    if config.scheduler == 'step':
        scheduler = StepLRScheduler(
            optimizer, decay_t=config.decay_epochs, decay_rate=config.lr_decay_factor,
            warmup_t=config.warmup_epochs, warmup_lr_init=config.warmup_lr
        )
    elif config.scheduler == 'plateau':
        scheduler = PlateauLRScheduler(
            optimizer, decay_rate=config.lr_decay_factor, patience_t=config.patience_epochs,
            verbose=False, threshold=1e-4, cooldown_t=0, mode='max',
            warmup_t=config.warmup_epochs, warmup_lr_init=config.warmup_lr, lr_min=config.min_lr
        )
    elif config.scheduler == 'cosine':
        factor = 1. if config.t_in_epochs else config.iters_per_epoch
        scheduler = CosineLRScheduler(
            optimizer, t_initial=(config.num_epochs - config.warmup_epochs)*factor, lr_min=config.min_lr,
            warmup_t=config.warmup_epochs*factor, warmup_lr_init=config.warmup_lr, warmup_prefix=True,
            t_in_epochs=config.t_in_epochs
        )
    else:
        raise NotImplementedError(f'Scheduler `{config.scheduler}` is not found!')

    return scheduler


def get_loss(config):
    """Return a loss.

    Parameters:
        config (argparse.Namespace): arguments

    Returns:
        nn.Module: a loss function
    """
    if config.task == 'ssl' and config.ssl_method == 'simsiam':
        loss = nn.CosineSimilarity(dim=1)
    elif (config.task == 'ssl' and config.ssl_method == 'byol') or config.task == 'reg':
        loss = nn.MSELoss()
    else:
        if config.label_smoothing > 0.:
            loss = LabelSmoothingCrossEntropy(smoothing=config.label_smoothing)
        else:
            loss = nn.CrossEntropyLoss()

    return loss
