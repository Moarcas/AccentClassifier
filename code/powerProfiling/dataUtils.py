import torch
from torch.utils.data import Dataset
import numpy as np
import os.path


class AudioDataset(Dataset):
    def __init__(self, data_filepath, labels_filepath, batch_size):
        self.data = np.load(data_filepath)
        if batch_size == 64:
            self.data = np.expand_dims(self.data, axis=1)
        self.labels = np.load(labels_filepath)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        return data, label


def save_batch(feature_type, batch_size):
    features_path = f'../../data/arrays/{feature_type}_features.npy'
    batch_file = f'../../data/arrays/{feature_type}_batch_{batch_size}.npy'
    labels_path = '../../data/arrays/labels.npy'

    dataset = AudioDataset(data_filepath=features_path,
                           labels_filepath=labels_path,
                           batch_size=batch_size)

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


def get_batch(feature_type, batch_size):
    if feature_type == 'amplitude':
        batch_file = f'../../data/arrays/amplitude_batch_{batch_size}.npy'
    elif feature_type == 'mfcc':
        batch_file = f'../../data/arrays/mfcc_batch_{batch_size}.npy'
    else:
        print('Invalid argument')
        return None

    if not os.path.isfile(batch_file):
        save_batch(feature_type, batch_size)

    return torch.from_numpy(np.load(batch_file))
