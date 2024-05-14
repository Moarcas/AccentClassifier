from tqdm import tqdm
import os
from glob import glob
import librosa
import numpy as np
from typing import Type

trimmedaudio_path = '../data/trimmedData/*.wav'
array_mfcc_file = '../data/arrays/mfcc.npy'
array_normalized_mfcc_file = '../data/arrays/normalized_mfcc.npy'
array_labels_file = '../data/arrays/labels/npy'

sample_rate = 22050

language_label_map = {
    "american": 0,
    "australian": 1,
    "bangla": 2,
    "british": 3,
    "indian": 4,
    "malayalam": 5,
    "odiya": 6,
    "telugu": 7,
    "welsh": 8,
}


def make_standard_length(filename: str, nr_seconds: int) -> Type[np.ndarray]:
    sig, _ = librosa.load(filename)
    sig = librosa.util.fix_length(data=sig, size=sample_rate * nr_seconds, mode='wrap')
    return sig


def get_mfcc(signal: np.ndarray) -> Type[np.ndarray]:
    mfcc_feat = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=13, n_fft=512, hop_length=128)
    return mfcc_feat.T


def get_label(filepath):
    language = os.path.basename(filepath).split('_')[0]
    return language_label_map[language]


def get_array_mfcc_labels(folder: str) -> (Type[np.ndarray], Type[np.ndarray]):
    lst_mfcc = []
    lst_labels = []
    for filename in tqdm(glob(trimmedaudio_path)):
        normed_signal = make_standard_length(filename, nr_seconds=4)
        lst_mfcc.append(get_mfcc(normed_signal))
        lst_labels.append(get_label(filename))
    array_mfcc = np.array(lst_mfcc)
    array_labels = np.array(lst_labels)
    return (array_mfcc, array_labels)


def normalize(m):
    means = np.mean(m, axis=1, keepdims=True)
    m -= means
    stds = np.std(m, axis=1, keepdims=True)
    m /= stds
    return m


def save_array_mfcc_labels():
    array_mfcc, array_labels = get_array_mfcc_labels(trimmedaudio_path)
    np.save(array_mfcc_file, array_mfcc)
    np.save(array_labels_file, array_labels)


def save_array_normalized_mfcc_labels():
    array_mfcc = np.load(array_mfcc_file)
    array_normalized_mfcc = normalize(array_mfcc)
    np.save(array_normalized_mfcc_file, array_normalized_mfcc)

# save_array_normalized_mfcc_labels()
