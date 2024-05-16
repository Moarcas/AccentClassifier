import torch
import sys
import numpy as np
from snntorch import spikegen


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

        self.spikes_path = f'../../data/arrays/{
            feature_type}_{conversion_type}_spikes.npy'
        self.features_path = f'../../data/arrays/{
            feature_type}_features.npy'

        self.num_steps = 5

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
        spikes = spikegen.latency(
            features, num_steps=self.num_steps, normalize=True, linear=True)
        return spikes

    def convert(self):
        """ Converts features to binary spikes

        Saves:
            spikes: a int8 numpy array of events of shape (num_samples, input_size, timestamps)
            labels: a int8 numpy array of events of shape (num_samples)
        """
        print(f'Converting {self.feature_type} to spikes using {
              self.conversion_type} ecoding...')

        features = np.load(self.features_path)
        spikes = self.convert2spikes(features)

        np.save(self.spikes_path, spikes)

        print('Spikes saved successfully in', self.spikes_path)


def main():
    available_feature = ["amplitude", "mel", "mfcc"]
    available_encoding = ["delta", "rate", "latency"]

    if len(sys.argv) != 3 or \
       sys.argv[1] not in available_feature or \
       sys.argv[2] not in available_encoding:
        print("Usage: python3 convert_to_spikes.py <feature> <encoding>")
        print("Please provide two arguments when running the script.")
        print("Available arguments for feature:", ", ".join(available_feature))
        print("Available arguments for encoding:",
              ", ".join(available_encoding))
        return

    selected_feature_type = sys.argv[1]
    selected_conversion_type = sys.argv[2]

    a2s = Audio2Spikes(conversion_type=selected_conversion_type,
                       feature_type=selected_feature_type,
                       audio_dir='../../data/trimmedData/*.wav')
    a2s.convert()


if __name__ == '__main__':
    main()
