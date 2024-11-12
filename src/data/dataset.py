import os

import pandas as pd
import torch
import torch.utils.data as data

from src.data.transforms import *


class CustomDataset(data.Dataset):
    """A universal Dataset class that can load several datasets in `.feather` and `.pt` format.

    This class loads the data file all at once and saves it in memory.
    Process the original dataset by the corresponding function in `preprocess.py` first.
    """
    def __init__(self, config, type, transform):
        super(CustomDataset, self).__init__()
        self.config = config
        self.type = type
        self.transform = transform
        # load dataset
        try:
            data_path = os.path.join(config.dataset_dir, config.dataset, f'{type}.pt')
            self.samples, self.targets = self.load_pt_dataset(data_path)
        except:
            data_path = os.path.join(config.dataset_dir, config.dataset, f'{type}.feather')
            self.samples, self.targets = self.load_feather_dataset(data_path)
        # process dataset
        self.process_dataset()
        # set dataset parameters
        self.num_channels = self.samples.shape[1]
        self.series_size = self.samples.shape[-1]
        self.num_classes = len(self.config.classes)
        # adjust transform_size
        if config.transform_size_mode == 'auto':
            auto_transform_size = (self.series_size // config.window_size + 1) * config.window_size
            self.transform.transforms[0].size = auto_transform_size
            config.transform_size = auto_transform_size
        # few shot
        if self.config.few_shot_learning:
            shot_num = self.config.trainset_shot_num if type == 'train' else self.config.testset_shot_num
            self.sample_few_shot_task(shot_num)

    def load_pt_dataset(self, data_path):
        data = torch.load(data_path)
        try:
            samples = data['samples'].float()  # torch tensor: (n_samples, n_features, length)
        except:
            samples = torch.Tensor(data['samples']).float()
        try:
            targets = data['labels'].long()  # torch tensor: (n_samples,)
        except:
            targets = torch.Tensor(data['labels']).long()  # torch tensor: (n_samples,)
        self.config.classes = [str(c.item()) for c in data['labels'].unique()]
        # shuffle for train
        if self.type == 'train':
            perm = torch.randperm(samples.shape[0])
            idx = perm[:int(self.config.sample_ratio * len(perm))]
            samples = samples[idx]
            targets = targets[idx]

        return samples, targets

    def load_feather_dataset(self, data_path):
        data = pd.read_feather(data_path)
        classes = sorted(data.iloc[:, 0].unique())
        self.config.classes = classes
        class_to_idx = {c: i for i, c in enumerate(classes)}
        data.iloc[:, 0] = data.iloc[:, 0].map(class_to_idx)
        if self.type == 'train':
            data = data.sample(frac=self.config.sample_ratio, random_state=42)
        samples = torch.Tensor(data.iloc[:, 1:].values).float()
        targets = torch.Tensor(data.iloc[:, 0].values).long()

        return samples, targets

    def process_dataset(self):
        if self.samples.ndim == 2:
            self.samples.unsqueeze_(1)  # (n_samples, n_features, length)
        return

    def sample_few_shot_task(self, shot_num):
        class_indices = [torch.where(self.targets == class_idx)[0]
                            for class_idx in range(self.num_classes)]
        sample_indices_list = []
        # random pick k-shot samples for each class
        for indices in class_indices:
            if len(indices) < shot_num:
                sample_indices_list.append(indices)
            else:
                sample_indices_list.append(indices[torch.randperm(len(indices))][:shot_num])
        sample_indices = torch.cat(sample_indices_list, dim=0)
        sample_indices = sample_indices[torch.randperm(len(sample_indices))]
        self.samples = self.samples[sample_indices]
        self.targets = self.targets[sample_indices]

    def __getitem__(self, index: int):
        """Get the sample at the given index."""
        sample = self.samples[index]
        target = self.targets[index]
        # get rid of NaN in the end of the series
        sample = sample[:, ~torch.all(sample.isnan(), dim=0)]
        # fill remaining NaN with zero
        sample = torch.nan_to_num(sample, nan=0.)
        # augmentation and resize
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

    def __len__(self):
        """Get the length of the dataset."""
        return self.samples.shape[0]
