import os
import numpy as np
from tensorflow.keras.utils import to_categorical
import sys


def make_seq_and_labels(label_map, DATA_PATH):
    sequences1, labels1 = [], []

    # Loop through every file in every directory
    for action in os.listdir(DATA_PATH):
        fd = os.path.join(DATA_PATH + "/" + action)
        for sequence in os.listdir(fd):
            window = []
            fd1 = os.path.join(DATA_PATH + "/" + action + "/" + sequence)
            for frame in os.listdir(fd1):
                res = np.load(os.path.join(DATA_PATH + "/" + action + "/" + sequence + "/" + frame))

                # Remove all inhomogeneous data
                if len(np.array(res, dtype=object).shape) == 1:
                    window.append(res)

            # Only accept 30 frames data
            if len(window) == 30:
                sequences1.append(window)
                labels1.append(label_map[action])
    return sequences1, labels1


if __name__ == "__main__":
    # Insert the current directory into the sys path
    sys.path.insert(0, './')

    # Define the data path and action labels
    PATH = os.path.join('DATA')
    actions = np.array(["Sitting", "Standing", "Walking"])
    label_m = {label: num for num, label in enumerate(actions)}
    sequences, labels = make_seq_and_labels(label_m, PATH)

    # Convert sequences and labels to numpy arrays
    X = np.array(sequences)
    y = to_categorical(labels).astype(int)

    with open('X.npy', 'wb') as f:
        np.save(f, X)
    with open('y.npy', 'wb') as f:
        np.save(f, y)
