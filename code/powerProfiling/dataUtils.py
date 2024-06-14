import torch
from torch.utils.data import Dataset
import numpy as np
import os.path


class AudioDataset(Dataset):
    def __init__(self, data_filepath, labels_filepath, batch_size, arhitecture):
        self.data = np.load(data_filepath)
        if arhitecture == 'cnn':
            self.data = np.expand_dims(self.data, axis=1)
        self.labels = np.load(labels_filepath)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        return data, label


def normalize(data):
    data = (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
    return data


def mean_variance_normalization(data):
    mean = np.mean(data, axis=0, keepdims=True)
    std = np.std(data, axis=0, keepdims=True)
    normalized_data = (data - mean) / std
    return normalized_data


def save_batch(feature_type, batch_size, arhitecture):
    features_path = f'../../data/arrays/{feature_type}_features.npy'
    batch_file = f'../../data/arrays/{feature_type}_batch_{batch_size}_{arhitecture}.npy'
    labels_path = '../../data/arrays/labels.npy'

    dataset = AudioDataset(data_filepath=features_path,
                           labels_filepath=labels_path,
                           batch_size=batch_size,
                           arhitecture=arhitecture)

    # split the dataset into three parts (train 70%, test 15%, validation 15%)
    test_size = 0.15
    val_size = 0.15

    test_amount, val_amount = int(dataset.__len__() * test_size), int(dataset.__len__() * val_size)

    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [
                (dataset.__len__() - (test_amount + val_amount)),
                test_amount,
                val_amount
    ])

    train_dataloader = torch.utils.data.DataLoader(
                train_set,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
    )

    batch = next(iter(train_dataloader))[0]
    np.save(batch_file, batch.detach().numpy())


def get_batch(feature_type, batch_size, arhitecture):
    if feature_type == 'amplitude':
        batch_file = f'../../data/arrays/amplitude_batch_{batch_size}_{arhitecture}.npy'
    elif feature_type == 'mfcc':
        batch_file = f'../../data/arrays/mfcc_batch_{batch_size}_{arhitecture}.npy'
    else:
        print('Invalid argument')
        return None

    if not os.path.isfile(batch_file):
        save_batch(feature_type, batch_size, arhitecture)

    return torch.from_numpy(np.load(batch_file))
