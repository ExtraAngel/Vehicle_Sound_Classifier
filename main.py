import numpy as np
import librosa as lb
import matplotlib.pyplot as plt
from os import listdir
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from sklearn import svm
#from sklearn import neighbors
import pickle


# CONSTANTS
SAMPLING_RATE = 48000
#TRAIN_RATE = 0.8
TEST_RATE = 0.2
SEED = 41


def readSamples(path):
    """
    Read audio samples from the given path.

    :param path: Path to the audio file folder.
    :return: Audio power spectrograms.
    """
    audios = []
    maxSize = 0

    counter = 0
    for file in listdir(path):
        if counter > 1100:
            break
        counter += 1

        audio, sr = lb.load(path + file, sr=None) #TODO: audioread

        # In case the SR is different from expected, resample
        if sr > SAMPLING_RATE:
            audio = lb.resample(audio, orig_sr=sr, target_sr=SAMPLING_RATE)

        spectrogram = np.abs(lb.stft(audio, dtype=np.float32)).T  # n_fft=2048, hop_length=512 #TODO: check dtype

        # Get the maximum size of the spectrogram among the samples
        if spectrogram.shape[0] > maxSize:
            maxSize = spectrogram.shape[0]

        audios.append(spectrogram ** 2)
    return audios, maxSize


def padSamples(samples, maxSize):
    """
    Pads the samples with zeroes so they all have the same shape

    :param samples: Samples to pad with zeroes
    :param maxSize: Maximum size of a given samples
    :return: Numpy array of padded samples with shape (maxSize, sample.shape[1])
    """
    samples = [np.pad(sample, ((0, maxSize - sample.shape[0]), (0, 0)), 'constant') for sample in samples]
    return np.array(samples)


def getMetrics(y_test, y_pred, label):
    """
    Returns the precision and recall score for the label

    :param y_test: Test labels/ground truth
    :param y_pred: Predicted labels
    :param label: The label to calculate metrics for
    :return: Precision and recall score of the label
    """
    precision = precision_score(y_test, y_pred, pos_label=label)
    recall = recall_score(y_test, y_pred, pos_label=label)
    return precision, recall


def main():
    # Load tram audio spectrogram's and their respective sampling rates:
    tramSamples, maxTramSize = readSamples("Samples/Tram/")
    print(f"Loaded {len(tramSamples)} tram samples.")

    # Load car audio spectrogram's and their respective sampling rates:
    carSamples, maxCarSize = readSamples("Samples/Car/")
    print(f"Loaded {len(carSamples)} car samples.")

    maxSize = max(maxTramSize, maxCarSize)

    # Pad the samples to have the same size:
    tramSamples = padSamples(tramSamples, maxSize)
    carSamples = padSamples(carSamples, maxSize)
    print(f"Padded samples to size: {maxSize}")

    # # Combine the samples into a single array:
    samples = np.concat((tramSamples, carSamples), axis=0, dtype=np.float32)
    samples = samples.reshape(samples.shape[0], -1 )
    print(f"Reshaped samples to size: {samples.shape}")

    # # Add labels to the samples:
    labels = np.concat((np.ones(len(tramSamples)), np.zeros(len(carSamples))), axis=0, dtype=np.float32)

    # Split into training and testing sets:
    X_train, X_test, y_train, y_test = train_test_split(samples, labels, test_size=TEST_RATE, random_state=SEED)
    print(f"Training set size: {X_train.shape[0]} samples.")
    print(f"Testing set size: {X_test.shape[0]} samples.")

    # Train the SVM model:
    #model = svm.SVC(C=1, kernel='sigmoid', random_state=SEED)
    model = svm.LinearSVC(random_state=SEED)
    print(f"Model starting training...")
    model.fit(X_train, y_train)
    print("Model trained.")

    # Save the model to a file
    with open('linear_svc_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    print("Model saved to 'svm_model.pkl'.")

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


if __name__ == "__main__":
    main()