from tqdm import tqdm
from collections import OrderedDict
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

        self.train_features_path = f'../../data/arrays/train_{feature_type}_features.npy'
        self.train_labels_path = '../../data/arrays/train_labels.npy'
        self.test_features_path = f'../../data/arrays/test_{feature_type}_features.npy'
        self.test_labels_path = '../../data/arrays/test_labels.npy'

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
        languages_with_one_speaker = ['odiya', 'welsh']
        speaker_for_test = '01'

        train_features_array = []
        train_labels_array = []
        test_features_array = []
        test_labels_array = []

        for filename in tqdm(glob(self.audio_dir)):
            language, speaker_number = filename.split('/')[4].split('_')[0:2]
            speaker_number = speaker_number[1:]

            audio, _ = librosa.load(filename)
            features = self.convert2features(audio)
            label = self.get_label(filename)

            if language in languages_with_one_speaker or speaker_number != speaker_for_test:
                # add to train
                train_features_array.append(features)
                train_labels_array.append(label)
            else:
                # add to test
                continue
                test_features_array.append(features)
                test_labels_array.append(label)

        train_features_array = np.stack(train_features_array, axis=0)
        train_labels_array = np.stack(train_labels_array, axis=0)
        #test_features_array = np.stack(test_features_array, axis=0)
        #test_labels_array = np.stack(test_labels_array, axis=0)

        np.save(self.train_features_path, train_features_array)
        np.save(self.train_labels_path, train_labels_array)
        #np.save(self.test_features_path, test_features_array)
        #np.save(self.test_labels_path, test_labels_array)

        print('Train features saved successfully in', self.train_features_path)
        print('Train labels saved successfully in', self.train_labels_path)

        print('Test features saved successfully in', self.test_features_path)
        print('Test labels saved successfully in', self.test_labels_path)


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
