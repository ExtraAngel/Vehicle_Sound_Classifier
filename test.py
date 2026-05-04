from os import listdir
import librosa as lb
from sklearn import svm
import numpy as np
import pickle
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
from main import getMetrics, getFeatures, SEED, TEST_RATE, FILE_NAME


def main():
    # Load the model:
    with open(FILE_NAME, "rb") as f:
        model = pickle.load(f)
    print(f"Model {FILE_NAME} loaded successfully.")

    # Load tram audio spectrogram's and their respective sampling rates:
    tramSamples = getFeatures("Samples/Tram/Train/")
    print(f"Loaded {len(tramSamples)} tram samples.")

    # Load car audio spectrogram's and their respective sampling rates:
    carSamples = getFeatures("Samples/Car/Train/")
    print(f"Loaded {len(carSamples)} car samples.")

    # Combine the samples into a single array:
    samples = np.concat((tramSamples, carSamples), axis=0, dtype=np.float32)
    samples = samples.reshape(samples.shape[0], -1)

    # Add labels to the samples:
    labels = np.concat((np.ones(len(tramSamples)), np.zeros(len(carSamples))), axis=0, dtype=np.float32)

    _, X_test, _, y_test = train_test_split(samples, labels, test_size=TEST_RATE, random_state=SEED)
    print(f"Testing set size: {X_test.shape[0]} samples.")

    predictions = model.predict(X_test)
    carPrecision, carRecall, accuracy = getMetrics(y_test, predictions, 0)
    tramPrecision, tramRecall, accuracy = getMetrics(y_test, predictions, 1)

    # Display the results:
    print(f"Metric Results based on all samples (Test Rate is {TEST_RATE}):")
    print(f"Model accuracy: {accuracy * 100:.2f}%")
    print(f"Model precision for car: {carPrecision * 100:.2f}%")
    print(f"Model recall for car: {carRecall * 100:.2f}%")
    print(f"Model precision for tram: {tramPrecision * 100:.2f}%")
    print(f"Model recall for tram: {tramRecall * 100:.2f}%")

if __name__ == "__main__":
    main()