import numpy as np
import librosa as lb
import matplotlib.pyplot as plt
import sounddevice as sd
from os import listdir
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import neighbors


# CONSTANTS
#TRAIN_RATE = 0.8
TEST_RATE = 0.2
SEED = 41


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
    maxSize = 0

    for file in listdir(path):
        audio, sr = lb.load(path + file, sr=None)
        spectogram = np.abs(lb.stft(audio))  # n_fft=2048, hop_length=512

        if spectogram.shape[0] > maxSize:
            maxSize = spectogram.shape[0]

        audios.append((spectogram ** 2, sr)) # TODO: remove sr when not needed

    return audios, maxSize


def padSamples(samples, maxSize):
    samples = [np.pad(sample[0], ((0, maxSize - sample[0].shape[0]), (0, 0)), 'constant') for sample in samples]
    return np.array(samples)


def getMetrics(y_test, y_pred, label):
    numTestLabel = len(y_test[y_test == label])
    numPredLabel = len(y_pred[y_pred == label])
    numLabel = len(y_test[y_test == y_pred])
    numPredCorrectLabel = ((numTestLabel + numPredLabel + numLabel) - len(y_test)) / 2

    precision = numPredCorrectLabel / numPredLabel
    recall = numPredCorrectLabel / numTestLabel
    return precision, recall


def main():
    # Load tram audio spectrogram's and their respective sampling rates:
    tramSamples, maxTramSize = readSamples("Samples/Tram/")

    # Load car audio spectrogram's and their respective sampling rates:
    carSamples, maxCarSize = readSamples("Samples/Car/")

    maxSize = max(maxTramSize, maxCarSize)

    # Pad the samples to have the same size:
    tramSamples = padSamples(tramSamples, maxSize)
    carSamples = padSamples(carSamples, maxSize)

    # Combine the samples into a single array:
    samples = np.concat((tramSamples, carSamples), axis=0)
    samples = samples.reshape(samples.shape[0], -1 )

    # Add labels to the samples:
    labels = np.concat((np.ones(len(tramSamples)), np.zeros(len(carSamples))), axis=0)

    # Split into training and testing sets:
    X_train, X_test, y_train, y_test = train_test_split(samples, labels, test_size=TEST_RATE, random_state=SEED)

    # Train the SVM model:
    model = svm.SVC(C=1, kernel='sigmoid', random_state=SEED)
    model.fit(X_train, y_train)

    # Evaluate the model:
    predictions = model.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    carPrecision, carRecall = getMetrics(y_test, predictions, 0)
    tramPrecision, tramRecall = getMetrics(y_test, predictions, 1)

    # Display the results:
    print(f"Model accuracy: {accuracy * 100:.2f}%")
    print(f"Model precision for car: {carPrecision * 100:.2f}%")
    print(f"Model recall for car: {carRecall * 100:.2f}%")
    print(f"Model precision for tram: {tramPrecision * 100:.2f}%")
    print(f"Model recall for tram: {tramRecall * 100:.2f}%")


    #lb.display.specshow(lb.power_to_db(tramSamples[0][0]), sr=tramSamples[0][1], x_axis='time', y_axis='linear')
    #plt.show()



if __name__ == "__main__":
    main()