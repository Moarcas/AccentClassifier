from glob import glob
import torch
import sys
from generate_arrays import make_standard_length, get_label
import numpy as np
import librosa
from tqdm import tqdm
from snntorch import spikegen
from sklearn import preprocessing


class Audio2Spikes:
    """The Audio2Spikes class manages the conversion from raw audio into spikes
       and stores the required conversion parameters.

    Attributes:
        conversion_type: the type of conversion used: delta, rate or latency
    """

    def __init__(self, conversion_type, feature_type, audio_dir):
        self.conversion_type = conversion_type
        self.feature_type = feature_type
        self.audio_dir = audio_dir

        if conversion_type == "delta":
            self.convert2spikes = self.convert2delta_2
        elif conversion_type == "rate":
            self.convert2spikes = self.convert2rate
        elif conversion_type == "latency":
            self.convert2spikes = self.convert2latency
        else:
            raise ValueError("Invalid conversion type")

        if feature_type == "amplitude":
            self.convert2features = self.identity
        elif feature_type == "mel":
            self.convert2features = self.convert2mel
        elif feature_type == "mfcc":
            self.convert2features = self.convert2mfcc
        else:
            raise ValueError("Invalid feature type")

        self.spikes_path = f'../data/arrays/{feature_type}_{conversion_type}_spikes.npy'
        self.labels_path = f'../data/arrays/{feature_type}_{conversion_type}_labels.npy'

        if conversion_type == "delta":
            self.standardize_function = lambda x: x
        else:
            self.standardize_function = self.std_fun

        self.std_fun = preprocessing.MinMaxScaler().fit_transform

        self.num_steps = 5

        self._default_spec_kwargs_mel = {
            "sr": 22050,
            "n_mels": 20,
            "n_fft": 512,
            "hop_length": 256,
        }

        self._default_spec_kwargs_mfcc = {
            "sr": 22050,
            "n_mfcc": 13,
            "n_mels": 20,
            "n_fft": 512,
            "hop_length": 256,
        }

    def std_fun(self, feature):
        feature = self.standardize_function(feature)
        feature = np.clip(feature, 0, 1)

    def identity(self, audio):
        audio = self.standardize_function(audio.reshape(-1, 1)).T
        return audio

    def convert2mel(self, audio):
        mel_spectogram = librosa.feature.melspectrogram(y=audio, **self._default_spec_kwargs_mel)
        mel_spectogram = np.log(mel_spectogram)
        mel_spectogram = self.standardize_function(mel_spectogram)
        return mel_spectogram

    def convert2mfcc(self, audio):
        mfcc = librosa.feature.mfcc(y=audio, **self._default_spec_kwargs_mfcc)
        mfcc = self.standardize_function(mfcc)
        return mfcc

    def convert2delta_2(self, features):
        features = np.expand_dims(features, 1)
        features = torch.from_numpy(features).permute(2, 1, 0)
        features = spikegen.delta(features)
        return features

    def convert2delta(self, features, threshold=1):
        """ Converts features to binary spikes via delta
            modulation (https://en.wikipedia.org/wiki/Delta_modulation).

        Args:
            audio: numpy array of shape (input_size, timestamps)
            threshold: The difference between the residual and signal that
            will be considered an increase or decrease. Defaults to 1.

        Returns:
            A int8 numpy array of events of shape (input_size, timestamps).
        """

        events = np.zeros(features.shape)
        levels = np.round(features[:, 0])

        for t in range(features.shape[-1]):
            events[:, t] = (features[:, t] - levels > threshold).astype(int) - (
                features[:, t] - levels < -threshold).astype(int)
            levels += events[:, t] * threshold

        return events

    def convert2rate(self, features):
        """ Converts features to binary spikes via rate encoding

        Args:
            audio: numpy array of shape (input_size, timestamps)

        Returns:
            A int8 numpy array of events of shape (input_size, timestamps).
        """

        features = torch.from_numpy(features)
        spikes = spikegen.rate(features, time_var_input=True)
        return spikes

    def convert2latency(self, features):
        """ Converts features to binary spikes via latency encoding

        Args:
            audio: numpy array of shape (input_size, timestamps)

        Returns:
            A int8 numpy array of events of shape (num_steps, input_size, timestamps).
        """

        features = torch.from_numpy(features)
        spikes = spikegen.latency(features, num_steps=self.num_steps, normalize=True, linear=True)
        return spikes

    def convert(self):
        """ Converts features to binary spikes

        Saves:
            spikes: a int8 numpy array of events of shape (num_samples, input_size, timestamps)
            labels: a int8 numpy array of events of shape (num_samples)
        """
        print(f'Converting {self.feature_type} to spikes using {self.conversion_type} ecoding...')
        spikes_array = []
        labels_array = []
        for filename in tqdm(glob(self.audio_dir)):
            audio = make_standard_length(filename, nr_seconds=4)
            features = self.convert2features(audio)

            label = get_label(filename)
            spikes = self.convert2spikes(features)

            spikes_array.append(spikes)
            labels_array.append(label)

        spikes_array = np.concatenate(spikes_array, axis=1)
        labels_array = np.stack(labels_array, axis=0)

        np.save(self.spikes_path, spikes_array)
        np.save(self.labels_path, labels_array)

        print('Spikes saved successfully in', self.spikes_path)
        print('Labels saved successfully in', self.labels_path)


def main():
    available_feature = ["amplitude", "mel", "mfcc"]
    available_encoding = ["delta", "rate", "latency"]

    if len(sys.argv) != 3 or \
       sys.argv[1] not in available_feature or \
       sys.argv[2] not in available_encoding:
        print("Usage: python3 convert_to_spikes.py <feature> <encoding>")
        print("Please provide two arguments when running the script.")
        print("Available arguments for feature:", ", ".join(available_feature))
        print("Available arguments for encoding:", ", ".join(available_encoding))
        return

    selected_feature_type = sys.argv[1]
    selected_conversion_type = sys.argv[2]

    a2s = Audio2Spikes(conversion_type=selected_conversion_type,
                       feature_type=selected_feature_type,
                       audio_dir='../data/trimmedData/*.wav')
    a2s.convert()


if __name__ == '__main__':
    main()
