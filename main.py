import numpy as np
import librosa as lb
from os import listdir
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn import svm
import pickle
import logging


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# CONSTANTS:
AUDIO_FEATURES = ["mfcc"]
KERNEL = "rbf"
N_MFCC = 4
N_FFT = 256
FILE_NAME = "model.pkl"
TEST_RATE = 0.5
SEED = 41


def getFeatures(path, audio_features=AUDIO_FEATURES):
    """
    Returns the features of the sounds found in path.
    The features are calculated, and their means will
    be added to the output for a more consistent shape

    :param path: str, Path to the audio file folder.
    :param audio_features: arr[str], The audio feature to use on the samples.
                          Options: "mfcc"(default), "zcr", "rms", "cqt"
    :return: Audio samples with their given features
    """
    audios = []

    for file in listdir(path):
        audio, sr = lb.load(path + file, sr=None)

        # Get the given audio features:
        feats = np.array([])
        if "mfcc" in audio_features:
            mfcc_mean = np.mean(lb.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC), axis=1)
            feats = np.hstack([feats, mfcc_mean])

        if "cqt"  in audio_features:
            cqt = lb.feature.chroma_cqt(y=audio, sr=sr).mean()
            feats = np.hstack([feats, cqt])

        if "rms" in audio_features:
            rms = lb.feature.rms(y=audio).mean()
            feats = np.hstack([feats, rms])

        if "zcr"  in audio_features:
            zcr = lb.feature.zero_crossing_rate(audio).mean()
            feats = np.hstack([feats, zcr])

        if "spectral" in audio_features:
            centroid = lb.feature.spectral_centroid(y=audio, sr=sr).mean()
            bandwidth = lb.feature.spectral_bandwidth(y=audio, sr=sr).mean()
            feats = np.hstack([feats, [centroid, bandwidth]])

        if "power"  in audio_features:
            power = np.mean(np.abs(lb.stft(y=audio, n_fft=N_FFT)) ** 2, axis=1)
            feats = np.hstack([feats, power])


        if feats.size == 0:
            raise Exception("Please enter a at least one valid audio feature "
                            "\"mfcc\", \"cqt\", \"rms\", \"zcr\" or \"spectral\".")

        audios.append(feats)
    return audios


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


def trainAndTest():
    # Load tram audio clips and their maximum sample size:
    tramTrainSamples = getFeatures("Samples/Tram/Train/")
    tramTestSamples = getFeatures("Samples/Tram/Test/")
    logging.info(f"Loaded {len(tramTrainSamples) + len(tramTestSamples)} tram samples.")

    # Load car audio clips and their maximum sample size
    carTrainSamples = getFeatures("Samples/Car/Train/")
    carTestSamples = getFeatures("Samples/Car/Test/")
    logging.info(f"Loaded {len(carTrainSamples) + len(carTestSamples)} car samples.")

    # Combine the samples into training and testing sets:
    X_train = np.concat((tramTrainSamples, carTrainSamples), axis=0, dtype=np.float32)
    y_train = np.concat((np.ones(len(tramTrainSamples)), np.zeros(len(carTrainSamples))), axis=0, dtype=np.float32)

    X_test = np.concat((tramTestSamples, carTestSamples), axis=0, dtype=np.float32)
    y_test = np.concat((np.ones(len(tramTestSamples)), np.zeros(len(carTestSamples))), axis=0, dtype=np.float32)

    # Testing on all the data samples, not just only our recordings
    # X = np.concat((X_train, X_test), axis=0)
    # y = np.concat((y_train, y_test), axis=0)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_RATE, random_state=SEED)

    logging.info(f"Training set size: {X_train.shape[0]} samples, with each sample being size {X_train.shape[1]}.")
    logging.info(f"Testing set size: {X_test.shape[0]} samples, with each sample being size {X_test.shape[1]}.")

    # Train the SVM model:
    model = svm.SVC(C=1, kernel=KERNEL, random_state=SEED)
    logging.info(f"Model starting training...")
    model.fit(X_train, y_train)
    logging.info("Model trained.")

    # Save the model to a file
    with open(FILE_NAME, 'wb') as model_file:
        pickle.dump(model, model_file)
    logging.info(f"Model saved to '{FILE_NAME}'.")

    # Evaluate the model:
    predictions = model.predict(X_test)
    carPrecision, carRecall, accuracy = getMetrics(y_test, predictions, 0)
    tramPrecision, tramRecall, accuracy = getMetrics(y_test, predictions, 1)

    # Display the results:
    logging.info(f"Model accuracy: {accuracy * 100:.2f}%")
    logging.info(f"Model precision for car: {carPrecision * 100:.2f}%")
    logging.info(f"Model recall for car: {carRecall * 100:.2f}%")
    logging.info(f"Model precision for tram: {tramPrecision * 100:.2f}%")
    logging.info(f"Model recall for tram: {tramRecall * 100:.2f}%")


def testOnly(Use50TestRate = False):
    # Load the model:
    with open(FILE_NAME, "rb") as f:
        model = pickle.load(f)
    logging.info(f"Model {FILE_NAME} loaded successfully.")

    if Use50TestRate:
        # Load tram audio spectrogram's and their respective sampling rates:
        tramSamples = getFeatures("Samples/Tram/Train/")
        logging.info(f"Loaded {len(tramSamples)} tram samples.")

        # Load car audio spectrogram's and their respective sampling rates:
        carSamples = getFeatures("Samples/Car/Train/")
        logging.info(f"Loaded {len(carSamples)} car samples.")

        # Combine the samples into a single array:
        samples = np.concat((tramSamples, carSamples), axis=0, dtype=np.float32)
        samples = samples.reshape(samples.shape[0], -1)

        # Add labels to the samples:
        labels = np.concat((np.ones(len(tramSamples)), np.zeros(len(carSamples))), axis=0, dtype=np.float32)

        _, X_test, _, y_test = train_test_split(samples, labels, test_size=TEST_RATE, random_state=SEED)
        logging.info(f"Testing set size: {X_test.shape[0]} samples.")

    # Only test on our own recordings:
    else:
        # Load the features:
        tramSamples = getFeatures("Samples/Tram/Test/")
        logging.info(f"Loaded {len(tramSamples)} tram samples.")

        carSamples = getFeatures("Samples/Car/Test/")
        logging.info(f"Loaded {len(carSamples)} car samples.")

        # Reshape them into test set:
        X_test = np.concat((tramSamples, carSamples), axis=0, dtype=np.float32)
        y_test = np.concat((np.ones(len(tramSamples)), np.zeros(len(carSamples))), axis=0, dtype=np.float32)

    predictions = model.predict(X_test)
    carPrecision, carRecall, accuracy = getMetrics(y_test, predictions, 0)
    tramPrecision, tramRecall, accuracy = getMetrics(y_test, predictions, 1)

    # Display the results:
    logging.info(f"Model accuracy: {accuracy * 100:.2f}%")
    logging.info(f"Model precision for car: {carPrecision * 100:.2f}%")
    logging.info(f"Model recall for car: {carRecall * 100:.2f}%")
    logging.info(f"Model precision for tram: {tramPrecision * 100:.2f}%")
    logging.info(f"Model recall for tram: {tramRecall * 100:.2f}%")


def main():
    # Uncomment the desired function to run:
    #trainAndTest()
    testOnly()


if __name__ == "__main__":
    main()