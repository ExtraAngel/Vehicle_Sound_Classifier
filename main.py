import numpy as np
import librosa as lb
from os import listdir
from sklearn.metrics import precision_score, recall_score
from sklearn import svm
import pickle


# CONSTANTS:
AUDIO_FEATURE = "mfcc"
FILE_NAME = "mfcc.pkl"
SAMPLING_RATE = 48000
TEST_RATE = 0.5
SEED = 41


# TODO: rename func
def readSamples(path, audio_feature=AUDIO_FEATURE):
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
            feat = lb.feature.mfcc(y=audio, sr=sr, n_mfcc=8)
        elif audio_feature == "rms":
            feat = lb.feature.rms(y=audio)
        elif audio_feature == "zcr":
            feat = lb.feature.zero_crossing_rate(y=audio)
        elif audio_feature == "cqt":
            feat = lb.feature.chroma_cqt(y=audio, sr=sr)
        else:
            raise Exception("Please enter a valid audio feature \"mfcc\", \"rms\", \"zcr\" or \"cqt\".")

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


def formatInputData(tramSamples, carSamples, maxSize):
    # Pad the samples to be the same shape:
    X_tram = padSamples(tramSamples, maxSize)
    X_car = padSamples(carSamples, maxSize)
    print(f"Padded samples to size: {maxSize}")

    # Combine the samples into a single array:
    X = np.concat((X_tram, X_car), axis=0, dtype=np.float32)
    X = X.reshape(X.shape[0], -1)
    print(f"Shape of samples: {X.shape}")

    # Create the labels
    y = np.concat((np.ones(len(tramSamples)), np.zeros(len(carSamples))), axis=0, dtype=np.float32)

    return X, y


def main():
    # Load tram audio clips and their maximum sample size:
    tramTrainSamples, maxTramTrainSize = readSamples("Samples/Tram/Train/")
    tramTestSamples, maxTramTestSize = readSamples("Samples/Tram/Test/")
    print(f"Loaded {len(tramTrainSamples) + len(tramTestSamples)} tram samples.")

    # Load car audio clips and their maximum sample size
    carTrainSamples, maxCarTrainSize = readSamples("Samples/Car/Train/")
    carTestSamples, maxCarTestSize = readSamples("Samples/Car/Test/")
    print(f"Loaded {len(carTrainSamples) + len(carTestSamples)} car samples.")

    maxSize = max(maxTramTrainSize, maxCarTrainSize, maxTramTestSize, maxCarTestSize)

    # Get the train and test data from samples in correct format:
    X_train, y_train = formatInputData(tramTrainSamples, carTrainSamples, maxSize)
    X_test, y_test = formatInputData(tramTestSamples, carTestSamples, maxSize)

    print(f"Training set size: {X_train.shape[0]} samples.")
    print(f"Testing set size: {X_test.shape[0]} samples.")

    # Train the SVM model:
    model = svm.SVC(C=1, kernel='linear', random_state=SEED)
    print(f"Model starting training...")
    model.fit(X_train, y_train)
    print("Model trained.")

    # Save the model to a file
    with open(FILE_NAME, 'wb') as model_file:
        pickle.dump(model, model_file)
    print(f"Model saved to '{FILE_NAME}'.")

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