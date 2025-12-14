from os import listdir
import librosa as lb
from sklearn import svm
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from main import getMetrics, padSamples, readSamples, SEED, TEST_RATE

def main():
    # load
    with open("linear_svc_model.pkl", "rb") as f:
        clf2 = pickle.load(f)


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
    samples = samples.reshape(samples.shape[0], -1)
    print(f"Reshaped samples to size: {samples.shape}")

    # # Add labels to the samples:
    labels = np.concat((np.ones(len(tramSamples)), np.zeros(len(carSamples))), axis=0, dtype=np.float32)

    _, X_test, _, y_test = train_test_split(samples, labels, test_size=TEST_RATE, random_state=SEED)

    predictions = clf2.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    carPrecision, carRecall = getMetrics(y_test, predictions, 0)
    tramPrecision, tramRecall = getMetrics(y_test, predictions, 1)

    # Display the results:
    print(f"Model accuracy: {accuracy * 100:.2f}%")
    print(f"Model precision for car: {carPrecision * 100:.2f}%")
    print(f"Model recall for car: {carRecall * 100:.2f}%")
    print(f"Model precision for tram: {tramPrecision * 100:.2f}%")
    print(f"Model recall for tram: {tramRecall * 100:.2f}%")

    pass

if __name__ == "__main__":
    main()