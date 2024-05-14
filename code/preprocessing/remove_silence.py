from pydub import AudioSegment
from tqdm import tqdm
import os
from glob import glob

def detect_leading_silence(sound, silence_threshold=-30.0, chunk_size=10):
    '''
    sound is a pydub.AudioSegment
    silence_threshold in dB
    chunk_size in ms

    iterate over chunks until you find the first one with sound
    '''
    trim_ms = 0 # ms

    assert chunk_size > 0 # to avoid infinite loop
    while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold and trim_ms < len(sound):
        trim_ms += chunk_size

    return trim_ms

rawaudio_path = '../data/rawdata/**/*.wav'
trimmedaudio_path = '../data/trimmedData/'

for filename in tqdm(glob(rawaudio_path, recursive=True)):
    sound = AudioSegment.from_file(filename, format="wav")

    start_trim = detect_leading_silence(sound)
    end_trim = detect_leading_silence(sound.reverse())

    duration = len(sound)
    trimmed_sound = sound[start_trim:duration-end_trim]
    trimmedfilename = os.path.join(trimmedaudio_path, os.path.basename(filename))
    trimmed_sound.export(trimmedfilename, format="wav")
