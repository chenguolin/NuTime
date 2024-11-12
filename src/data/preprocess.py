import os
import glob
import random

from tqdm import tqdm
import numpy as np
import pandas as pd
import mne
from scipy.io.arff import loadarff
from sklearn.model_selection import train_test_split
import torch


def prepareUCR(dataset_dir: str, fill_missing_and_variable=True, normalize=False, test_ratio=0.3, grain=10000):
    """Prepare UCR sub-datasets and merge them into one united dataset.

    Download the original dataset archive from https://www.cs.ucr.edu/~eamonn/time_series_data_2018/ and
    extract it in the `dataset_dir` directory.

    Parameters:
        dataset_dir (str): directory of UCR archive
        fill_missing_and_variable (bool): whether to use repaired values provided by UCR-128 archive
        normalize (bool): whether to normalize sub-datasets by its mean and std when merging
        test_ratio (float): test dataset ratio
        grain (int): grain size of splitting the huge merged dataset into multiple files
    """
    assert os.path.exists(dataset_dir), f'UCR archive directory `{dataset_dir}` does not exist!'

    # delete these sub-datasets to merge the whole UCR archive without duplication
    DEL_DATASETS = [
        'Chinatown', 'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect', 
        'DodgerLoopGame', 'DodgerLoopWeekend', 'FreezerSmallTrain', 'GesturePebbleZ2',
        'GunPointMaleVersusFemale', 'GunPointOldVersusYoung', 'InsectEPGSmallTrain',
        'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxOutlineCorrect', 'MixedShapesSmallTrain',
        'PhalangesOutlinesCorrect', 'PickupGestureWiimoteZ', 'ProximalPhalanxOutlineAgeGroup',
        'ProximalPhalanxOutlineCorrect', 'SemgHandGenderCh2', 'SemgHandSubjectCh2',
        'ShakeGestureWiimoteZ', 'WormsTwoClass'
    ]

    root = os.path.abspath(os.path.join(dataset_dir, '..'))
    save_dir = os.path.join(root, 'ucr-128')
    merge_dir, merge_split_dir = os.path.join(root, 'ucr-merged'), os.path.join(root, 'ucr-merged-split')

    MVVL_DIR = 'Missing_value_and_variable_length_datasets_adjusted'
    MVVL_DATASETS = os.listdir(os.path.join(dataset_dir, MVVL_DIR))

    DATASETS = os.listdir(dataset_dir)
    DATASETS.remove(MVVL_DIR)
    assert len(DATASETS) == 128, f'UCR archive directory `{dataset_dir}` should contain 128 sub-datasets, but found {len(DATASETS)}!'

    all_data = pd.DataFrame()  # split data into train and test randomly
    train_split, test_split = pd.DataFrame(), pd.DataFrame()  # split data into train and test by the original dataset

    for d in tqdm(sorted(DATASETS)):
        if d in MVVL_DATASETS and fill_missing_and_variable:
            # missing values are treated with linear interpolation
            # variable length are treated with low-amplitude noise padding
            df_train = pd.read_csv(os.path.join(dataset_dir, MVVL_DIR, d, f'{d}_TRAIN.tsv'), sep='\t', header=None)
            df_test = pd.read_csv(os.path.join(dataset_dir, MVVL_DIR, d, f'{d}_TEST.tsv'), sep='\t', header=None)
        else:
            df_train = pd.read_csv(os.path.join(dataset_dir, d, f'{d}_TRAIN.tsv'), sep='\t', header=None)
            df_test = pd.read_csv(os.path.join(dataset_dir, d, f'{d}_TEST.tsv'), sep='\t', header=None)

        # prepare data in sub-datasets, i.e., change category names and file suffix (.tsv -> .csv)
        data = pd.concat([df_train, df_test], axis=0)
        if d in ['FordA', 'FordB']:  # merge categories in the two datasets
            new_idx = {sorted(data.iloc[:, 0].unique())[i]: f'Ford_{i:02d}' for i in range(len(data.iloc[:, 0].unique()))}
        else:
            new_idx = {sorted(data.iloc[:, 0].unique())[i]: f'{d}_{i:02d}' for i in range(len(data.iloc[:, 0].unique()))}
        data.iloc[:, 0] = data.iloc[:, 0].map(new_idx)
        df_train.iloc[:, 0] = df_train.iloc[:, 0].map(new_idx)
        df_test.iloc[:, 0] = df_test.iloc[:, 0].map(new_idx)

        os.makedirs(os.path.join(save_dir, d), exist_ok=True)
        df_train.columns, df_test.columns = [str(i) for i in range(df_train.shape[1])], [str(i) for i in range(df_test.shape[1])]
        df_train.reset_index(drop=True).to_feather(os.path.join(save_dir, d, f'train.feather'))
        df_test.reset_index(drop=True).to_feather(os.path.join(save_dir, d, f'test.feather'))

        # merge sub-datasets
        if d not in DEL_DATASETS:
            if normalize:  # normalize each sub-dataset by its mean and std of the whole dataset
                data_mean, data_std = data.iloc[:, 1:].values.mean(), data.iloc[:, 1:].values.std()
                data.iloc[:, 1:] = (data.iloc[:, 1:] - data_mean) / (data_std + 1e-8)
            all_data = pd.concat([all_data, data], axis=0, sort=True)
            if normalize:  # normalize each sub-dataset by its mean and std of the training dataset for the split version
                train_mean, train_std = df_train.iloc[:, 1:].values.mean(), df_train.iloc[:, 1:].values.std()
                df_train.iloc[:, 1:] = (df_train.iloc[:, 1:] - train_mean) / (train_std + 1e-8)
                df_test.iloc[:, 1:] = (df_test.iloc[:, 1:] - train_mean) / (train_std + 1e-8)
            train_split, test_split = pd.concat([train_split, df_train], axis=0, sort=True), pd.concat([test_split, df_test], axis=0, sort=True)

    # randomly split data into train and test
    samples_train, samples_test, targets_train, targets_test = train_test_split(
        all_data.iloc[:, 1:], all_data.iloc[:, 0],
        test_size=test_ratio,
        random_state=42, shuffle=True,
        stratify=all_data.iloc[:, 0]
    )

    train = pd.concat([targets_train, samples_train], axis=1)
    test = pd.concat([targets_test, samples_test], axis=1)

    print(f'Saving processed data: {len(train)}/{len(train_split)} samples in train/train_split set; {len(test)}/{len(test_split)} samples in test/test_split set')
    os.makedirs(merge_dir, exist_ok=True)
    os.makedirs(merge_split_dir, exist_ok=True)
    # feather file requires column names in string format and drop index
    train.columns, test.columns = [str(i) for i in range(train.shape[1])], [str(i) for i in range(test.shape[1])]
    train_split.columns, test_split.columns = [str(i) for i in range(train_split.shape[1])], [str(i) for i in range(test_split.shape[1])]

    # random split data into train and test
    train.sample(frac=1, random_state=42)  # shuffle the data
    test.sample(frac=1, random_state=42)  # shuffle the data
    for i in tqdm(range(train.shape[0]//grain + 1)):
        end = min((i+1)*grain, train.shape[0])
        train.iloc[i*grain:end].reset_index(drop=True).to_feather(os.path.join(merge_dir, f'train_{i}.feather'))
    for i in tqdm(range(test.shape[0]//grain + 1)):
        end = min((i+1)*grain, test.shape[0])
        test.iloc[i*grain:end].reset_index(drop=True).to_feather(os.path.join(merge_dir, f'test_{i}.feather'))
    train.reset_index(drop=True).to_feather(os.path.join(merge_dir, 'train.feather'))
    test.reset_index(drop=True).to_feather(os.path.join(merge_dir, 'test.feather'))

    # split data into train and test followed the original archive
    train_split.sample(frac=1, random_state=42)  # shuffle the data
    test_split.sample(frac=1, random_state=42)  # shuffle the data
    for i in tqdm(range(train_split.shape[0]//grain + 1)):
        end = min((i+1)*grain, train_split.shape[0])
        train_split.iloc[i*grain:end].reset_index(drop=True).to_feather(os.path.join(merge_split_dir, f'train_{i}.feather'))
    for i in tqdm(range(test_split.shape[0]//grain + 1)):
        end = min((i+1)*grain, test_split.shape[0])
        test_split.iloc[i*grain:end].reset_index(drop=True).to_feather(os.path.join(merge_split_dir, f'test_{i}.feather'))
    train_split.reset_index(drop=True).to_feather(os.path.join(merge_split_dir, 'train.feather'))
    test_split.reset_index(drop=True).to_feather(os.path.join(merge_split_dir, 'test.feather'))


def prepareUEA(dataset_dir: str, normalize=False, grain=10000):
    """Prepare UEA sub-datasets and merge them into one unlabeled dataset.

    Download the original dataset archive from http://www.timeseriesclassification.com/dataset.php and
    extract it in the `dataset_dir` directory.

    Parameters:
        dataset_dir (str): directory of UEA archive
        normalize (bool): whether to normalize sub-datasets by its mean and std when merging
        grain (int): grain size of splitting the huge merged dataset into multiple files
    """
    assert os.path.exists(dataset_dir), f'UEA archive directory `{dataset_dir}` does not exist!'

    DATASETS = os.listdir(dataset_dir)
    DATASETS = list(set(DATASETS) - set(['Descriptions', 'Images', 'DataDimensions.csv', 'UEAArchive2018.pdf']))  # remove unnecessary files
    assert len(DATASETS) == 30, f'UEA archive directory `{dataset_dir}` should contain 30 sub-datasets, but found {len(DATASETS)}!'

    root = os.path.abspath(os.path.join(dataset_dir, '..'))
    save_dir, merge_dir = os.path.join(root, 'uea-30'), os.path.join(root, 'uea-merged')

    all_samples = pd.DataFrame()
    for d in tqdm(sorted(DATASETS)):
        train = loadarff(os.path.join(dataset_dir, d, f'{d}_TRAIN.arff'))[0]
        samples, labels = [], []
        for sample, label in train:  # instance-loop
            sample = np.array([s.tolist() for s in sample])  # (d, l)
            label = label.decode("utf-8")  # (1,)
            samples.append(sample)
            labels.append(label)
        samples = torch.tensor(samples, dtype=torch.float)  # (n, d, l)
        classes = set(labels)  # unique labels
        class_to_idx = {c: i for i, c in enumerate(classes)}  # map label (string) to index (int)
        labels = torch.tensor([class_to_idx[l] for l in labels], dtype=torch.long)  # (n,)
        data = {'samples': samples, 'labels': labels}

        test = loadarff(os.path.join(dataset_dir, d, f'{d}_TEST.arff'))[0]
        samples_test, labels_test = [], []
        for sample, label in test:  # instance-loop
            sample = np.array([s.tolist() for s in sample])  # (d, l)
            label = label.decode("utf-8")  # (1,)
            samples_test.append(sample)
            labels_test.append(label)
        samples_test = torch.tensor(samples_test, dtype=torch.float)  # (n, d, l)
        labels_test = torch.tensor([class_to_idx[l] for l in labels_test], dtype=torch.long)  # (n,)
        data_test = {'samples': samples_test, 'labels': labels_test}

        # save multi-dimensional data as `.pt` file
        os.makedirs(os.path.join(save_dir, d), exist_ok=True)
        torch.save(data, os.path.join(save_dir, d, f'train.pt'))
        torch.save(data_test, os.path.join(save_dir, d, f'test.pt'))

        if d not in ['EigenWorms', 'FaceDetection', 'InsectWingbeat', 'PenDigits']:  # do not merge these datasets for some reasons
            if normalize:  # normalize data by training set mean and std per channel
                train_mean, train_std = samples.mean(dim=(0, 2)), samples.std(dim=(0, 2))
                if torch.isnan(train_mean).any() or torch.isnan(train_std).any():  # NaN in the dataset
                    train_mean, train_std = np.nanmean(samples.numpy(), axis=(0, 2)), np.nanstd(samples.numpy(), axis=(0, 2))
                train_mean = torch.as_tensor(train_mean, dtype=torch.float).view(-1, 1)
                train_std = torch.as_tensor(train_std, dtype=torch.float).view(-1, 1)
                samples = (samples - train_mean) / (train_std + 1e-8)
                samples_test = (samples_test - train_mean) / (train_std + 1e-8)

            samples_flat = samples.reshape(-1, samples.shape[-1]).numpy()  # (n*d, l)
            samples_test_flat = samples_test.reshape(-1, samples_test.shape[-1]).numpy()  # (n*d, l)
            all_samples = pd.concat([all_samples, pd.DataFrame(samples_flat), pd.DataFrame(samples_test_flat)])

    # save uni-dimensional data as `.feather` file
    os.makedirs(os.path.join(merge_dir), exist_ok=True)
    all_samples.columns = [str(i) for i in range(all_samples.shape[1])]
    all_samples.sample(frac=1, random_state=42)  # shuffle the data
    for i in tqdm(range(all_samples.shape[0]//grain + 1)):
        end = min((i+1)*grain, all_samples.shape[0])
        all_samples.iloc[i*grain:end].reset_index(drop=True).to_feather(os.path.join(merge_dir, f'train_{i}.feather'))
    all_samples.reset_index(drop=True).to_feather(os.path.join(merge_dir, 'train.feather'))


def prepareSleepEDF(dataset_dir: str, subdataset_name='age', channel='EEG Fpz-Cz', num_records=39, split_level='subject', test_ratio=0.2, grain=10000):
    """Prepare Sleep-EDF dataset.

    Download the original dataset from https://www.physionet.org/content/sleep-edfx/1.0.0/ and
    extract it in the `dataset_dir` directory.

    Parameters:
        dataset_dir (str): directory of Sleep-EDF dataset
        subdataset_name (str): name of subdataset to be processed (`age` or `temazepam`)
        channel (str): channel to be processed (`EEG Fpz-Cz` or `EEG Pz-Oz`)
        num_records (int | None): number of records to be processed (default: 39, i.e., the first 20 subjects included in the older version of the dataset)
        split_level (str): level of split (`subject` or `record`)
        test_ratio (float): test dataset ratio
        grain (int): grain size of splitting the huge merged dataset into multiple files
    """
    assert os.path.exists(dataset_dir), f'Sleep-EDF directory `{dataset_dir}` does not exist!'
    assert subdataset_name in ['age', 'temazepam'], f'Parameter `subdataset_name` must be one of `age` or `temazepam`, but got {subdataset_name}!'
    if subdataset_name == 'age':
        assert channel in ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal'], \
            'Parameter `channel` must be one of `EEG Fpz-Cz`, `EEG Pz-Oz` or `EOG horizontal`, when `subdataset_name` is `age`!'
    else:  # subdataset_name == 'temazepam'
        assert channel in ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'EMG submental'], \
            'Parameter `channel` must be one of `EEG Fpz-Cz`, `EEG Pz-Oz`, `EOG horizontal`, or `EMG submental`, when `subdataset_name` is `temazepam`!'

    root = os.path.abspath(os.path.join(dataset_dir, '..'))
    name = 'sleep-edf'
    if subdataset_name != 'age': name += f'_temazepam'
    if channel != 'EEG Fpz-Cz': name += f'_{channel.lower().replace(" ", "_")}'
    if num_records is None: name += '_all'
    save_dir = os.path.join(root, name)

    save_record_dir = os.path.join(save_dir, 'records')
    os.makedirs(save_record_dir, exist_ok=True)
    save_subject_dir = os.path.join(save_dir, 'subjects')
    os.makedirs(save_subject_dir, exist_ok=True)

    if subdataset_name == 'age':
        dataset_dir = os.path.join(dataset_dir, 'sleep-cassette')
    else:  # if subdataset_name == 'temazepam'
        dataset_dir = os.path.join(dataset_dir, 'sleep-telemetry')

    psg_fnames = sorted(glob.glob(os.path.join(dataset_dir, "*PSG.edf")))
    ann_fnames = sorted(glob.glob(os.path.join(dataset_dir, "*Hypnogram.edf")))
    assert len(psg_fnames) == len(ann_fnames), f'Number of PSG and annotation files do not match: {len(psg_fnames)} vs. {len(ann_fnames)}!'
    if num_records is None:
        num_records = len(psg_fnames)
    else:  # if num_records is not None
        assert num_records <= len(psg_fnames), f'Parameter `num_records` must be less than or equal to {len(psg_fnames)}, but got {num_records}!'

    # subject-level split
    subjects, records, prev_sub_id = [], [], ''
    labels = {  # follow AASM standard: merge S3 and S4 stages into a single N3 stage
        'Sleep stage W':f'{name}_0', 'Sleep stage 1':f'{name}_1', 'Sleep stage 2':f'{name}_2',
        'Sleep stage 3':f'{name}_3', 'Sleep stage 4':f'{name}_3', 'Sleep stage R':f'{name}_4',
    }
    for i in tqdm(range(num_records)):
        # merge the records for the same subject
        sub_id = psg_fnames[i].split('/')[-1][3:5]  # e.g., '07' for 'SC4071E0-PSG.edf'
        if split_level == 'subject' and sub_id != prev_sub_id:  # a new subject
            # save one subject
            if len(records) > 0:
                subject = pd.concat(records, axis=0)  # merge all records for the same subject
                subject.to_csv(os.path.join(save_subject_dir, f'subject_{prev_sub_id}.csv'), header=False, index=False)
                subjects.append(subject)
            records, prev_sub_id = [], sub_id

        # read PSG and annotation files
        raw_data = mne.io.read_raw_edf(psg_fnames[i], preload=True)
        annotations = mne.read_annotations(ann_fnames[i])
        raw_data.set_annotations(annotations, emit_warning=False)

        # keep last 30-min wake events before sleep and first 30-min wake events after sleep, and redefine annotations on raw data
        for i in range(len(annotations)):
            if annotations[i]['description'][12] in ['1', '2', '3', '4', 'R']: break  # i: first non-weak stage
        for j in range(len(annotations)-1, -1, -1):
            if annotations[j]['description'][12] in ['1', '2', '3', '4', 'R']: break  # j: last non-weak stage
        annotations.crop(annotations[i]['onset'] - 30*60, annotations[j+1]['onset'] + 30*60)
        raw_data.set_annotations(annotations, emit_warning=False)

        try:
            # keep only interested (sleep) events
            event_id = {'Sleep stage W':0, 'Sleep stage 1':1, 'Sleep stage 2':2, 'Sleep stage 3':3, 'Sleep stage 4':4, 'Sleep stage R':5}
            events, _ = mne.events_from_annotations(raw_data, event_id=event_id, chunk_duration=30.)
            # split PSG data into several 30-s epochs; sample rate is 100Hz, so 30*100 = 3000 samples per epoch
            tmax = 30. - 1./raw_data.info['sfreq']  # tmax in included
            epochs = mne.Epochs(raw=raw_data, events=events, event_id=event_id, tmin=0., tmax=tmax, baseline=None, preload=True)
        except ValueError:  # if sleep stage 4 are not found
            try:
                event_id = {'Sleep stage W':0, 'Sleep stage 1':1, 'Sleep stage 2':2, 'Sleep stage 3':3, 'Sleep stage R':5}
                events, _ = mne.events_from_annotations(raw_data, event_id=event_id, chunk_duration=30.)
                tmax = 30. - 1./raw_data.info['sfreq']
                epochs = mne.Epochs(raw=raw_data, events=events, event_id=event_id, tmin=0., tmax=tmax, baseline=None, preload=True)
            except ValueError:  # if sleep stage 3 are not found
                event_id = {'Sleep stage W':0, 'Sleep stage 1':1, 'Sleep stage 2':2, 'Sleep stage R':5}
                events, _ = mne.events_from_annotations(raw_data, event_id=event_id, chunk_duration=30.)
                tmax = 30. - 1./raw_data.info['sfreq']
                epochs = mne.Epochs(raw=raw_data, events=events, event_id=event_id, tmin=0., tmax=tmax, baseline=None, preload=True)

        df = epochs.to_data_frame().loc[:, ['condition', 'epoch', channel]]
        df['timestamp'] = np.tile(np.arange(3000), len(df)//3000)

        # convert `channel` data in each epoch into one row and concatenate these epochs as one record
        samples = pd.pivot(df, values=channel, columns='timestamp', index='epoch')  # (#epochs, 3000)
        targets = pd.pivot(df, values='condition', columns='timestamp', index='epoch').iloc[:, 0:1]  # (#epochs, 1)
        record = pd.concat([targets, samples], axis=1)  # (#epochs, 3001)
        # save one record
        record_id = os.path.basename(psg_fnames[i]).split('-')[0]
        record.replace(labels, inplace=True)
        record.to_csv(os.path.join(save_record_dir, f'{i:03d}_{record_id}.csv'), header=False, index=False)
        # add this record to the list
        records.append(record)

    # save the last subject
    if len(records) > 0:
        subject = pd.concat(records, axis=0)  # merge all records for the same subject
        subject.to_csv(os.path.join(save_subject_dir, f'subject_{prev_sub_id}.csv'), header=False, index=False)
        subjects.append(subject)

    if num_records is None:
        if subdataset_name == 'age':
            assert len(records) == 153, f'Number of records in Sleep-EDF_sleep-cassette is not 153, but {len(records)}!'
        else:  # if subdataset_name == 'temazepam'
            assert len(records) == 44, f'Number of records in Sleep-EDF_sleep-telemetry is not 44, but {len(records)}!'

    random.seed(42)
    if split_level == 'subject':
        random.shuffle(subjects)
        train_subjects, test_subjects = subjects[:int(len(subjects)*(1-test_ratio) + 0.5)], subjects[int(len(subjects)*(1-test_ratio) + 0.5):]
        train_df, test_df = pd.concat(train_subjects, axis=0), pd.concat(test_subjects, axis=0)
    else:  # if split_level == 'record'
        random.shuffle(records)
        train_records, test_records = records[:int(len(records)*(1-test_ratio) + 0.5)], records[int(len(records)*(1-test_ratio) + 0.5):]
        train_df, test_df = pd.concat(train_records, axis=0), pd.concat(test_records, axis=0)
    # feather file requires column names in string format and drop index
    train_df.columns, test_df.columns = [str(i) for i in range(train_df.shape[1])], [str(i) for i in range(test_df.shape[1])]
    train_df.sample(frac=1, random_state=42)  # shuffle the data
    test_df.sample(frac=1, random_state=42)  # shuffle the data
    for i in tqdm(range(train_df.shape[0]//grain + 1)):
        end = min((i+1)*grain, train_df.shape[0])
        train_df.iloc[i*grain:end].reset_index(drop=True).to_feather(os.path.join(save_dir, f'train_{i}.feather'))
    for i in tqdm(range(test_df.shape[0]//grain + 1)):
        end = min((i+1)*grain, test_df.shape[0])
        test_df.iloc[i*grain:end].reset_index(drop=True).to_feather(os.path.join(save_dir, f'test_{i}.feather'))
    train_df.reset_index(drop=True).to_feather(os.path.join(save_dir, 'train.feather'))
    test_df.reset_index(drop=True).to_feather(os.path.join(save_dir, 'test.feather'))


def prepareEpilepsy(dataset_dir: str, val_ratio=0.1, test_ratio=0.1, binary_class=False):
    """Prepare Epilepsy Serizure dataset.

    Download the original dataset from https://raw.githubusercontent.com/emadeldeen24/TS-TCC/main/data_preprocessing/epilepsy/data_files/data.csv and
    move it to the `dataset_dir` directory.
    The original dataset is at https://archive.ics.uci.edu/ml/datasets/Epileptic+Seizure+Recognition, which is recently removed for some reason.

    Parameters:
        dataset_dir (str): directory of Epilepsy Serizure dataset
        test_ratio (float): test dataset ratio
    """
    assert os.path.exists(dataset_dir), f'Epilepsy Serizure directory `{dataset_dir}` does not exist!'

    root = os.path.abspath(os.path.join(dataset_dir, '..'))
    save_dir = os.path.join(root, 'epilepsy') if not binary_class else os.path.join(root, 'epilepsy_binary')

    # load data
    data = pd.read_csv(os.path.join(dataset_dir, 'data.csv'), index_col=0)

    # split data into train, val and test
    samples_train, samples_test, targets_train, targets_test = train_test_split(
        data.iloc[:, :-1], data.iloc[:, -1],
        test_size=val_ratio+test_ratio,
        random_state=42, shuffle=True,
        stratify=data.iloc[:, -1]
    )
    samples_val, samples_test, targets_val, targets_test = train_test_split(
        samples_test, targets_test,
        test_size=test_ratio/(val_ratio + test_ratio),
        random_state=42, shuffle=True
    )
    if binary_class:
        labels = {1:'epilepsy_binary_0', 2:'epilepsy_binary_1', 3:'epilepsy_binary_1', 4:'epilepsy_binary_1', 5:'epilepsy_binary_1'}
    else:  # keep the original labels
        labels = {1:'epilepsy_0', 2:'epilepsy_1', 3:'epilepsy_2', 4:'epilepsy_3', 5:'epilepsy_4'}
    targets_train, targets_val, targets_test = targets_train.replace(labels), targets_val.replace(labels), targets_test.replace(labels)
    train = pd.concat([targets_train, samples_train], axis=1)
    val = pd.concat([targets_val, samples_val], axis=1)
    test = pd.concat([targets_test, samples_test], axis=1)

    # reset feather columns and drop index
    train.columns, val.columns, test.columns = [str(i) for i in range(train.shape[1])], [str(i) for i in range(val.shape[1])], [str(i) for i in range(test.shape[1])]
    os.makedirs(save_dir, exist_ok=True)
    train.reset_index(drop=True).to_feather(os.path.join(save_dir, 'train.feather'))
    val.reset_index(drop=True).to_feather(os.path.join(save_dir, 'val.feather'))
    test.reset_index(drop=True).to_feather(os.path.join(save_dir, 'test.feather'))


def load_pt_dataset(data_path):
    data = torch.load(data_path)
    try:
        samples = data['samples'].float()  # torch tensor: (n_samples, n_features, length)
    except:
        samples = torch.Tensor(data['samples']).float()
    try:
        targets = data['labels'].long()  # torch tensor: (n_samples,)
    except:
        targets = torch.Tensor(data['labels']).long()  # torch tensor: (n_samples,)

    return samples, targets


def load_feather_dataset(data_path):
    data = pd.read_feather(data_path)
    classes = sorted(data.iloc[:, 0].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    data[data.columns[0]] = data[data.columns[0]].map(class_to_idx)

    samples = torch.Tensor(data.iloc[:, 1:].values).float()
    targets = torch.Tensor(data.iloc[:, 0].values).long()

    if samples.ndim == 2:
        samples.unsqueeze_(1)

    return samples, targets


def mergeDatasets(dataset_dir_dict: dict, save_dir: str, grain=10000):
    """merge provided datasets with original split of train test set.

    Parameters:
        dataset_dir_dict (dict): dictionery of the directory of source sub-datasets
        save_dir (str): save directory of the merged dataset
        grain (int): grain size of splitting the huge merged dataset into multiple files
    """
    # get sub-dataset dir list
    UEA_DIR_NAME = 'uea-30'
    UEA_DEL_DATASETS = ['InsectWingbeat', 'EigenWorms']
    # UEA_DEL_DATASETS = []
    MAX_LENGTH_LIMIT = 2048
    DATASETS = []
    merge_dir, merge_shuffle_dir = os.path.join(save_dir, 'all-merged-ori'), os.path.join(save_dir, 'all-merged')
    os.makedirs(merge_dir, exist_ok=True)
    os.makedirs(merge_shuffle_dir, exist_ok=True)
    for directory, dataset_type in dataset_dir_dict.items():
        if dataset_type == 'single':
            DATASETS.append(directory)
        else:
            for d in os.listdir(directory):
                if directory.split('/')[-1] == UEA_DIR_NAME and d in UEA_DEL_DATASETS:
                    continue
                DATASETS.append(os.path.join(directory, d))
    
    # load sub-datasets
    merged_df = {}
    df_data = {}
    df_no_duplicate_data = []
    for t in ['train', 'val', 'test']:
        df_data[t] = []
    for d in tqdm(sorted(DATASETS)):
        archieve_name = d.split("/")[-2].split("-")[0]
        dataset_name = d.split("/")[-1]
        uniq_name = archieve_name + "-" + dataset_name
        data_dict = {}
        df_data_dict = {}
        val_set_exist = True
        for t in ['train', 'val', 'test']:
            # load data
            try:
                try:
                    samples, targets = load_pt_dataset(os.path.join(d, f'{t}.pt'))
                except:
                    samples, targets = load_feather_dataset(os.path.join(d, f'{t}.feather'))
            except:
                # load test set for dataset without val set
                val_set_exist = False
                try:
                    samples, targets = load_pt_dataset(os.path.join(d, f'test.pt'))
                except:
                    samples, targets = load_feather_dataset(os.path.join(d, f'test.feather'))
            # split multi-variate sample to uni-variate samples
            N, C, L = samples.shape
            samples = samples.reshape(N*C, L)
            if L > MAX_LENGTH_LIMIT:
                samples = samples[:, :L]
            targets = np.repeat(targets, C, axis=0)
            data_dict[t] = {'samples': pd.DataFrame(samples.numpy()), 'targets': pd.DataFrame(targets.numpy())}
        # # remove large single datasets and long sequence 
        # if L > MAX_LENGTH_LIMIT:
        #     continue
        # reindex classes
        ori_idx = data_dict['train']['targets'].iloc[:, 0].unique()
        new_idx = {sorted(ori_idx)[i]: f'{uniq_name}_{i:02d}' for i in range(len(ori_idx))}
        # concat target and feature
        for t in ['train', 'val', 'test']:
            data_dict[t]['targets'].iloc[:, 0] = data_dict[t]['targets'].iloc[:, 0].map(new_idx)
            df_data_dict[t] = pd.concat([data_dict[t]['targets'], data_dict[t]['samples']], axis=1)
            df_data_dict[t].columns = [str(i) for i in range(df_data_dict[t].shape[1])]
            df_data[t].append(df_data_dict[t])
            if t != 'val' or val_set_exist:
                df_no_duplicate_data.append(df_data_dict[t])
        print(f'Finish process dataset:{uniq_name} '
                f'with train/val/test size: {df_data_dict["train"].shape[0]}/{df_data_dict["val"].shape[0]}/{df_data_dict["test"].shape[0]}, '
                f'features num: {C}, '
                f'class num: {len(new_idx)}, '
                f'length: {df_data_dict["train"].shape[1]}')

    # merge all datasets
    # for t in ['train', 'val', 'test']:
    for t in ['train']:
        merged_df[t] = pd.concat(df_data[t], axis=0)    # sort=True will reorder the sequence
        merged_df[t].columns = [str(i) for i in range(merged_df[t].shape[1])]
    print(f'Saving processed data: '
            f'{len(merged_df["train"])} samples in train set, '
            # f'{len(merged_df["val"])} samples in val set, '
            # f'{len(merged_df["test"])} samples in test set, '
            f'max_length: {merged_df["train"].shape[1]}, '
            f'total_classes: {len(merged_df["train"].iloc[:, 0].unique())}')

    # save original split data
    # for t in ['train', 'val', 'test']:
    for t in ['train']:
        for i in tqdm(range(merged_df[t].shape[0]//grain + 1)):
            end = min((i+1)*grain, merged_df[t].shape[0])
            merged_df[t].iloc[i*grain:end].reset_index(drop=True).to_feather(os.path.join(merge_dir, f'{t}_{i}.feather'))
        print(f'Start save whole {t} merged data')
        merged_df[t].reset_index(drop=True).to_feather(os.path.join(merge_dir, f'{t}.feather'))

    # save shuffled data
    # shuffled_df = {}
    # test_ratio = 0.3
    # print("Start merge data")
    # no_duplicate_merged_data = pd.concat(df_no_duplicate_data, axis=0)
    # no_duplicate_merged_data.columns = [str(i) for i in range(no_duplicate_merged_data.shape[1])]
    # print(f"Start shuffle merged data, "
    #         f"total_num: {no_duplicate_merged_data.shape[0]}, "
    #         f"classes: {len(no_duplicate_merged_data.iloc[:, 0].unique())}")
    # # shuffled_df['train'] = no_duplicate_merged_data.sample(frac=1.0, random_state=42)
    # shuffled_df['train'] = no_duplicate_merged_data
    # shuffled_df['test'] = no_duplicate_merged_data.sample(frac=test_ratio, random_state=42)
    # print(f'Saving shuffled data: '
    #         f'{len(shuffled_df["train"])} samples in train set, '
    #         f'{len(shuffled_df["test"])} samples in test set, '
    #         f'train total classes: {len(shuffled_df["train"].iloc[:, 0].unique())}, '
    #         f'test total classes: {len(shuffled_df["test"].iloc[:, 0].unique())}')
    # for t in ['train', 'test']:
    #     for i in tqdm(range(shuffled_df[t].shape[0]//grain + 1)):
    #         end = min((i+1)*grain, shuffled_df[t].shape[0])
    #         shuffled_df[t].iloc[i*grain:end].reset_index(drop=True).to_feather(os.path.join(merge_shuffle_dir, f'{t}_{i}.feather'))
    #     print(f'Start save whole {t} shuffle merged data')
    #     shuffled_df[t].reset_index(drop=True).to_feather(os.path.join(merge_shuffle_dir, f'{t}.feather'))


def prepareAnomalyDataset(data_dir, save_dir, seed=42, normal_num=900, outliers_num=100, test_ratio=0.3):
    samples, targets = load_pt_dataset(os.path.join(data_dir, 'test.pt'))
    undamaged_samples = samples[targets == 0]
    damaged_samples = samples[targets != 0]
    # sample normal
    perm1 = torch.randperm(undamaged_samples.shape[0])
    normal = undamaged_samples[perm1[:normal_num]]
    # sample outliers
    perm2 = torch.randperm(damaged_samples.shape[0])
    outliers = damaged_samples[perm2[:outliers_num]]
    # print(undamaged_samples.shape, damaged_samples.shape, normal.shape, outliers.shape)
    # shuffle
    new_samples = torch.cat((normal, outliers), 0)
    new_targets = torch.cat((torch.zeros(normal_num), torch.ones(outliers_num)), 0)
    perm = torch.randperm(new_samples.shape[0])
    new_samples = new_samples[perm]
    new_targets = new_targets[perm]
    # split
    train_num = int((normal_num + outliers_num) * (1 - test_ratio))
    train_samples = new_samples[:train_num]
    train_targets = new_targets[:train_num]
    test_samples = new_samples[train_num:]
    test_targets = new_targets[train_num:]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    obj = {'samples': train_samples, 'labels': train_targets}
    torch.save(obj, f'{save_dir}/train.pt')
    obj = {'samples': test_samples, 'labels': test_targets}
    torch.save(obj, f'{save_dir}/test.pt')


if __name__ == "__main__":
    # prepareEpilepsy("./datasets/time_series/epilepsy_origin")
    # prepareUCR(dataset_dir="./datasets/time_series/UCRArchive_2018",
    #             fill_missing_and_variable=True, normalize=False, test_ratio=0.3, grain=10000)
    # prepareUEA(dataset_dir="./datasets/time_series/Multivariate_arff")
    # prepareSleepEDF(dataset_dir="./datasets/time_series/physionet.org/files/sleep-edfx/1.0.0")

    # merge all datasets
    # dataset_dir_dict = {
    #     './datasets/time_series/uea-30': 'multi',
    #     './datasets/time_series/ucr-128': 'multi',
    #     './datasets/time_series/TFC-datasets': 'multi',
    # }
    # save_dir = './datasets/time_series/'
    # mergeDatasets(dataset_dir_dict, save_dir, grain=100000)

    # construct anomaly detection for FD-B
    for s in range(1, 6):
        prepareAnomalyDataset(
            data_dir="./datasets/time_series/TFC-datasets/FD-B",
            save_dir=f"./datasets/time_series/AnomalyDetection/FD-B/s{s}",
            seed=s)
