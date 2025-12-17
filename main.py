import numpy as np
import librosa as lb
from librosa import feature
import matplotlib.pyplot as plt
from os import listdir
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from sklearn import svm
import pickle


# CONSTANTS
SAMPLING_RATE = 48000
#TRAIN_RATE = 0.8
TEST_RATE = 0.2
SEED = 41


def readSamples(path, audio_feature="mfcc"):
    """
    Read audio samples from the given path.

    :param path: Path to the audio file folder.
    :param audio_feature: The audio feature to use on the samples.
                          Options: "mfcc"(default), "zcr", "rms", "cqt"
    :return: Audio samples with their given features
    """
    audios = []
    maxSize = 0

    counter = 0
    for file in listdir(path):
        if counter > 3000:
            break
        counter += 1

        audio, sr = lb.load(path + file, sr=None) #TODO: audioread

        # In case the SR is different from expected, resample
        if sr > SAMPLING_RATE:
            audio = lb.resample(audio, orig_sr=sr, target_sr=SAMPLING_RATE)

        # Load the given audio feature
        if audio_feature == "mfcc":
            feat = feature.mfcc(y=audio, sr=sr)
        elif audio_feature == "rms":
            feat = feature.rms(y=audio)
        elif audio_feature == "zcr":
            feat = feature.zero_crossing_rate(y=audio)
        elif audio_feature == "cqt":
            feat = feature.chroma_cqt(y=audio, sr=sr)
        else:
            raise Exception("Please enter a valid audio feature \"mfcc\", \"rms\", \"zcr\" or \"cqt\".")

        maxSize = max(maxSize, feat.shape[1])

        audios.append(feat)
    return audios, maxSize


def padSamples(samples, maxSize):
    """
    Pads the samples with zeroes so they all have the same shape

    :param samples: Samples to pad with zeroes
    :param maxSize: Maximum size of a given samples
    :return: Numpy array of padded samples with shape (sample.shape[0], maxSize)
    """
    samples = [np.pad(sample, ((0, 0), (0, maxSize - sample.shape[1])), 'constant') for sample in samples]
    return np.array(samples)


def getMetrics(y_test, y_pred, label):
    """
    Returns the precision, recall and accuracy score for the label

    :param y_test: Test labels/ground truth
    :param y_pred: Predicted labels
    :param label: The label to calculate metrics for
    :return: Precision, recall and accuracy score of the label
    """
    precision = precision_score(y_test, y_pred, pos_label=label)
    recall = recall_score(y_test, y_pred, pos_label=label)
    accuracy = np.mean(y_test == y_pred)
    return precision, recall, accuracy


def main():
    audio_feature = "mfcc"
    # Load tram audio features and their respective sampling rates:
    tramSamples, maxTramSize = readSamples("Samples/Tram/", audio_feature)
    print(f"Loaded {len(tramSamples)} tram samples.")

    # Load car audio features and their respective sampling rates:
    carSamples, maxCarSize = readSamples("Samples/Car/", audio_feature)
    print(f"Loaded {len(carSamples)} car samples.")

    maxSize = max(maxTramSize, maxCarSize)

    tramSamples = padSamples(tramSamples, maxSize)
    carSamples = padSamples(carSamples, maxSize)
    print(f"Padded samples to size: {maxSize}")

    # # Combine the samples into a single array:
    samples = np.concat((tramSamples, carSamples), axis=0, dtype=np.float32)
    samples = samples.reshape(samples.shape[0], -1)
    print(f"Shape of samples: {samples.shape}")

    labels = np.concat((np.ones(len(tramSamples)), np.zeros(len(carSamples))), axis=0, dtype=np.float32)

    X_train, X_test, y_train, y_test = train_test_split(samples, labels, test_size=TEST_RATE, random_state=SEED)
    print(f"Training set size: {X_train.shape[0]} samples.")
    print(f"Testing set size: {X_test.shape[0]} samples.")

    # Train the SVM model:
    model = svm.SVC(C=1, kernel='linear', random_state=SEED)
    print(f"Model starting training...")
    model.fit(X_train, y_train)
    print("Model trained.")

    # Save the model to a file
    with open('mfcc.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    print("Model saved to 'svm_model.pkl'.")

    # Evaluate the model:
    predictions = model.predict(X_test)
    carPrecision, carRecall, accuracy = getMetrics(y_test, predictions, 0)
    tramPrecision, tramRecall, accuracy = getMetrics(y_test, predictions, 1)

    # Display the results:
    print(f"Model accuracy: {accuracy * 100:.2f}%")
    print(f"Model precision for car: {carPrecision * 100:.2f}%")
    print(f"Model recall for car: {carRecall * 100:.2f}%")
    print(f"Model precision for tram: {tramPrecision * 100:.2f}%")
    print(f"Model recall for tram: {tramRecall * 100:.2f}%")


if __name__ == "__main__":
    main()