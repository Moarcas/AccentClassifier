import librosa
from pydub import AudioSegment
from tqdm import tqdm
import os
from glob import glob
import sys
import soundfile as sf


class AudioProcessor():
    def __init__(self, number_seconds):
        self.rawaudio_path = '../../data/rawdata/**/*.wav'
        self.trimmedaudio_path = '../../data/trimmedData/'
        self.number_seconds = number_seconds
        self.silence_threshold = -30.0
        self.chunk_size = 10
        self.sample_rate = 22050

    def detect_leading_silence(self, sound):
        '''
        sound is a pydub.AudioSegment
        silence_threshold in dB
        chunk_size in ms

        iterate over chunks until you find the first one with sound
        '''
        trim_ms = 0

        assert self.chunk_size > 0 # to avoid infinite loop
        while sound[trim_ms:trim_ms+self.chunk_size].dBFS < self.silence_threshold and trim_ms < len(sound):
            trim_ms += self.chunk_size
        return trim_ms

    def convert(self):
        for filename in tqdm(glob(self.rawaudio_path, recursive=True)):
            trimmedfilename = os.path.join(self.trimmedaudio_path,
                                           os.path.basename(filename))
            audio = AudioSegment.from_file(filename, format="wav")

            start_trim = self.detect_leading_silence(audio)
            end_trim = self.detect_leading_silence(audio.reverse())

            duration = len(audio)
            trimmed_sound = audio[start_trim:duration-end_trim]
            trimmedfilename = os.path.join(self.trimmedaudio_path,
                                           os.path.basename(filename))
            trimmed_sound.export(trimmedfilename, format="wav")

            trimmed_sound, _ = librosa.load(trimmedfilename)
            trimmed_sound = librosa.util.fix_length(data=trimmed_sound,
                                                    size=self.sample_rate *
                                                    self.number_seconds,
                                                    mode='wrap')
            sf.write(trimmedfilename, trimmed_sound, self.sample_rate, format='wav')


def main():
    if len(sys.argv) != 2 or \
       not sys.argv[1].isdigit() or \
       int(sys.argv[1]) < 1 or \
       int(sys.argv[1]) > 5:
        print("Usage: python3 processAudio.py <number seconds>")
        print("Please specify the number of seconds you want the recordings to have")
        print("The number of seconds must be between 1 and 5")
        return

    selected_number_seconds = int(sys.argv[1])

    audioProcessor = AudioProcessor(number_seconds=selected_number_seconds)
    audioProcessor.convert()


if __name__ == '__main__':
    main()
