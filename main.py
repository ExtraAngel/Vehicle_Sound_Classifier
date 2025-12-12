import numpy as np
import librosa as lb
import matplotlib.pyplot as plt
import sounddevice as sd
from os import listdir
#from sklearn import svm


def readSamples(path):
    """
    # TODO: fix docstring
    Read audio samples from the given path.

    Parameters:
    path (str): Path to the audio file.

    Returns:
    tuple: Audio power spectrograms and sampling rate.
    """
    audios = []

    for file in listdir(path):
        audio, sr = lb.load(path + file, sr=None)
        spectogram = np.abs(lb.stft(audio))  # n_fft=2048, hop_length=512

        audios.append((spectogram ** 2, sr))

    return audios


def main():
    # Load tram audio spectrogram's and their respective sampling rates:
    tramSamples = readSamples("Samples/Tram/")

    # Load car audio spectrogram's and their respective sampling rates:
    carSamples = readSamples("Samples/Car/")

    lb.display.specshow(lb.amplitude_to_db(tramSamples[0][0]), sr=tramSamples[0][1], x_axis='time', y_axis='linear')
    plt.show()


if __name__ == "__main__":
    main()