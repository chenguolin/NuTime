import functools

from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.dataset import *
from src.data.transforms import *


def get_dataset_cls(config):
    pt_keywords = ['uea-', 'TFC-']
    # pt_keywords = ['uea-', 'FD-', '-TFC', 'HAR', 'ECG', 'SleepEEG', 'UWaveGestureLibrary', 'EMG', 'waveform', 'waveform-nonorm']
    data_format = 'feather'
    for kw in pt_keywords:
        if kw in config.dataset or kw in config.dataset_dir:
            data_format = 'pt'
            break
    config.data_format = data_format
    dataset_cls = functools.partial(CustomDataset)

    return dataset_cls, data_format


def get_transform(config):
    transform_dict = {}
    if config.transform_type == 'rrcrop':
        # augmentation for train
        train_transform = TSCompose([
            TSRandomResizedCrop(size=config.transform_size, scale=config.scale, mode=config.interpolate_mode),
            TSRandomMask(scale=config.mask_scale, mask_mode=config.mask_mode,
                        win_size=config.window_size, window_mask_generator=config.window_mask_generator),
        ])
        # resize for test
        test_transform = TSCompose([
            TSResize(size=config.transform_size, mode=config.interpolate_mode),
        ])
    else:   # config.transform_type == 'identity'
        train_transform = TSIdentity()
        test_transform = TSIdentity()
    
    # get contrasive samples for self-supervised learning
    if config.task == 'ssl':
        train_transform = TSTwoViewsTransform(train_transform)

    transform_dict['train'] = train_transform
    transform_dict['val'] = test_transform
    transform_dict['test'] = test_transform

    return transform_dict


def normalize_dataset(config, dataset_dict):
    eps = 1e-8
    if config.norm == 'global':
        samples = dataset_dict['train'].samples
        mean, std = np.nanmean(samples.numpy(), axis=(0, 2)), np.nanstd(samples.numpy(), axis=(0, 2))
        mean = torch.as_tensor(mean).view(-1, 1)
        std = torch.as_tensor(std).view(-1, 1)
        for type in config.dataset_type_list:
            dataset_dict[type].samples = (dataset_dict[type].samples - mean) / (std + eps)
    elif config.norm == 'instance':
        num_channels = dataset_dict['train'].num_channels
        for type in config.dataset_type_list:
            mean, std = np.nanmean(dataset_dict[type].samples.numpy(), axis=(2)), np.nanstd(dataset_dict[type].samples.numpy(), axis=(2))
            mean = torch.as_tensor(mean).view(-1, num_channels, 1)
            std = torch.as_tensor(std).view(-1, num_channels, 1)
            dataset_dict[type].samples = (dataset_dict[type].samples - mean) / (std + eps)


def get_dataset(config):
    # dataset_cls, data_format = get_dataset_cls(config)
    transform_dict = get_transform(config)
    dataset_dict = {}
    for type in ['train', 'test']:
        dataset_dict[type] = CustomDataset(config=config, type=type, transform=transform_dict[type])
    # no validation set for some datasets
    try:
        config.no_validation_set = False
        dataset_dict['val'] = CustomDataset(config=config, type='val', transform=transform_dict['val'])
    except:
        config.no_validation_set = True
        dataset_dict['val'] = CustomDataset(config=config, type='test', transform=transform_dict['test'])
    # return train with no augmentation for knn evaluation
    if config.task == 'ssl' and config.use_eval:
        dataset_dict['train_knn'] = CustomDataset(config=config, type='train', transform=transform_dict['test'])
    config.dataset_type_list = dataset_dict.keys()
    normalize_dataset(config, dataset_dict)

    return dataset_dict


def get_weighted_sampler(config, dataset):
    class_counts = torch.bincount(dataset.targets)
    sample_weights = 1 / class_counts[dataset.targets]

    return WeightedRandomSampler(weights=sample_weights, num_samples=len(dataset), replacement=True)


def build_dataloader(config, dataset_dict):
    dataloader_dict = {}
    for type in config.dataset_type_list:
        sampler = None
        if config.use_weighted_sampler and type == 'train':
            sampler = get_weighted_sampler(config, dataset_dict[type])
        dataloader_dict[type] = DataLoader(
            dataset=dataset_dict[type],
            shuffle=True if type == 'train' and sampler is None else False,
            # shuffle=False,
            batch_size=config.batch_size if type == 'train' else config.eval_batch_size,
            num_workers=config.num_workers,
            sampler = sampler
        )

    return dataloader_dict


def get_dataloader(config):
    # get dataset
    dataset_dict = get_dataset(config)
    # get dataloader
    dataloader_dict = build_dataloader(config, dataset_dict)
    # set config
    config.ori_series_size = dataset_dict['train'].series_size
    config.model_series_size = config.transform_size
    config.num_classes = dataset_dict['train'].num_classes
    config.num_channels = dataset_dict['train'].num_channels
    config.iters_per_epoch = len(dataloader_dict['train'])

    return dataloader_dict
