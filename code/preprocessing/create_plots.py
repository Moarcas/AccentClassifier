from glob import glob
import librosa
from tqdm import tqdm
import seaborn
import wave
import contextlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def get_audio_length(fname):
    with contextlib.closing(wave.open(fname, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
    return duration


def plot_length_distribution(audio_lengths, save_path=None):
    plt.hist(audio_lengths, bins=20, edgecolor='black')
    plt.xlabel('Length (seconds)')
    plt.ylabel('Count')
    plt.title('Audio Length Distribution')
    if save_path:
        plt.savefig(save_path)
    plt.show()


def get_lengths_list(audio_path):
    audio_lengths = []
    for filename in glob(audio_path, recursive=True):
        length = get_audio_length(filename)
        audio_lengths.append(length)
    return audio_lengths


def plot_length_distribution_rawdata():
    audio_path = '../data/rawdata/**/*.wav'
    plot_path = '../plots/rawAudioLengths'
    lengths = get_lengths_list(audio_path)
    plot_length_distribution(lengths, plot_path)


def plot_length_distribution_trimmeddata():
    audio_path = '../data/trimmedData/*.wav'
    plot_path = '../plots/trimmedAudioLengths'
    lengths = get_lengths_list(audio_path)
    plot_length_distribution(lengths, plot_path)


def plot_languages_distribution():
    labels_path = '../data/arrays/labels.npy'
    plot_path = '../plots/languages'
    label_language_map = {
        0: "american",
        1: "australian",
        2: "bangla",
        3: "british",
        4: "indian",
        5: "malayalam",
        6: "odiya",
        7: "telugu",
        8: "welsh",
    }

    labels = np.load(labels_path)
    labels, counts = np.unique(labels, return_counts=True)
    languages = [label_language_map[label] for label in labels]

    plt.figure(figsize=(10, 6))
    plt.bar(languages, counts, color='skyblue')
    plt.xlabel('Languages')
    plt.ylabel('Counts')
    plt.title('Distribution of Languages')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.show()


def get_PCA(mfcc_array):
    mfcc_array = np.reshape(mfcc_array, (mfcc_array.shape[0], -1))
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(mfcc_array)

    # Normalize PCA components
    pca_result_normalized = StandardScaler().fit_transform(pca_result)

    return pca_result_normalized


def plot_PCA():
    plot_path = '../plots/pca'
    label_language_map = {
        0: "american",
        1: "australian",
        2: "bangla",
        3: "british",
        4: "indian",
        5: "malayalam",
        6: "odiya",
        7: "telugu",
        8: "welsh",
    }

    max_samples_language = 300

    array_mfcc_file = '../data/arrays/mfcc.npy'
    array_labels_file = '../data/arrays/labels.npy'

    mfcc_array = np.load(array_mfcc_file)
    labels_array = np.load(array_labels_file)

    pca_result = get_PCA(mfcc_array)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for label, language in label_language_map.items():
        indices = np.where(labels_array == label)[0][:max_samples_language]
        ax.scatter(pca_result[indices, 0], pca_result[indices, 1], pca_result[indices, 2], label=language)

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.legend()
    plt.savefig(plot_path)
    plt.show()


def get_mfcc_distributions(m):
    n_coeffs = m.shape[2]
    fig, axs = plt.subplots(n_coeffs, figsize=(25, 6), sharex=True, sharey=True)

    for coeff_no in tqdm(range(n_coeffs)):
        ax = axs[coeff_no]
        p = seaborn.color_palette("husl", m.shape[2])
        color = p[coeff_no]
        seaborn.histplot(m[:, :, coeff_no].flatten(), color=color, label=f'MFCC{coeff_no}', alpha=0.5, ax=ax, bins=20)

        ax.set(yticklabels=[])
        ax.set(ylabel=None)
        ax.tick_params(right=False)
        ax.get_yaxis().set_visible(False)
        seaborn.despine(ax=ax, left=True)

        ax.axvline(0.0, ls='--', alpha=0.5, color='black')

        ax.legend(loc='upper right')

        if coeff_no != (n_coeffs-1):
            ax.get_xaxis().set_visible(False)
            seaborn.despine(ax=ax, bottom=True, left=True)

    return fig


def plot_mfcc_distributions():
    plot_path_mfcc = '../plots/mfcc_distributions'
    plot_path_normalized_mfcc = '../plots/normalized_mfcc_distributions'
    mfcc_file = '../data/arrays/mfcc.npy'
    normalized_mfcc_file = '../data/arrays/normalized_mfcc.npy'

    mfcc = np.load(mfcc_file)
    normalized_mfcc = np.load(normalized_mfcc_file)

    fig = get_mfcc_distributions(mfcc)
    fig.axes[0].set_xlim(-600, 600)
    fig.suptitle("Original")
    plt.savefig(plot_path_mfcc)

    fig = get_mfcc_distributions(normalized_mfcc)
    fig.axes[0].set_xlim(-6, 6)
    fig.suptitle("Per-coefficient standardization")
    plt.savefig(plot_path_normalized_mfcc)
    plt.show()


def plot_audio_signal():
    plot_path_audio_signal = '../../plots/audio_signal'
    audio_file = '../../data/trimmeddata/american_s01_001.wav'
    signal, sr = librosa.load(audio_file)

    plt.figure()
    plt.plot(signal)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.savefig(plot_path_audio_signal)
    plt.plot()
    plt.show()


def plot_audio_mfcc():
    plot_path_audio_signal = '../../plots/audio_spectogram'
    audio_file = '../../data/trimmeddata/american_s01_001.wav'
    signal, sr = librosa.load(audio_file)


# plot_length_distribution_trimmeddata()
# plot_length_distribution_rawdata()
# plot_languages_distribution()
# plot_PCA()
#plot_mfcc_distributions()
plot_audio_signal()
