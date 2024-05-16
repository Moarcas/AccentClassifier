from tqdm import tqdm
import sys
import librosa
import os
from glob import glob
import numpy as np


class Audio2Features:
    """The Audio2Spikes class manages the conversion from raw audio into features
       and stores the required conversion parameters.

    Attributes:
        feature_type: the type of conversion used: mel, mfcc
    """

    def __init__(self, feature_type, audio_dir):
        self.feature_type = feature_type
        self.audio_dir = audio_dir

        if feature_type == "amplitude":
            self.convert2features = self.downsample
        elif feature_type == "mel":
            self.convert2features = self.convert2mel
        elif feature_type == "mfcc":
            self.convert2features = self.convert2mfcc
        else:
            raise ValueError("Invalid feature type")

        self.features_path = f'../../data/arrays/{feature_type}_features.npy'
        self.labels_path = '../../data/arrays/labels.npy'

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

        self.language_label_map = {
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

        self.sample_rate = 22050

    def downsample(self, audio):
        audio = librosa.resample(y=audio, orig_sr=self.sample_rate, target_sr=16000)
        return audio

    def get_label(self, filepath):
        language = os.path.basename(filepath).split('_')[0]
        return self.language_label_map[language]

    def convert2mel(self, audio):
        mel_spectogram = librosa.feature.melspectrogram(y=audio, **self._default_spec_kwargs_mel)
        mel_spectogram = np.log(mel_spectogram)
        return mel_spectogram

    def convert2mfcc(self, audio):
        mfcc = librosa.feature.mfcc(y=audio, **self._default_spec_kwargs_mfcc)
        return mfcc

    def convert(self):
        """ Converts features to features
        Saves:
            features: a int8 numpy array of events of shape (num_samples, input_size, timestamps)
            labels: a int8 numpy array of events of shape (num_samples)
        """
        print(f'Converting audio to {self.feature_type}...')

        features_array = []
        labels_array = []

        for filename in tqdm(glob(self.audio_dir)):
            audio, _ = librosa.load(filename)
            features = self.convert2features(audio)
            label = self.get_label(filename)

            features_array.append(features)
            labels_array.append(label)

        features_array = np.stack(features_array, axis=0)
        labels_array = np.stack(labels_array, axis=0)

        np.save(self.features_path, features_array)
        np.save(self.labels_path, labels_array)

        print('Features saved successfully in', self.features_path)
        print('Labels saved successfully in', self.labels_path)


def main():
    available_feature = ["amplitude", "mel", "mfcc"]

    if len(sys.argv) != 2 or \
       sys.argv[1] not in available_feature:
        print("Usage: python3 convert_to_spikes.py <feature>")
        print("Please provide one argument when running the script.")
        print("Available arguments for feature:", ", ".join(available_feature))
        return

    selected_feature_type = sys.argv[1]

    a2f = Audio2Features(feature_type=selected_feature_type,
                         audio_dir='../../data/trimmedData/*.wav')
    a2f.convert()


if __name__ == '__main__':
    main()
