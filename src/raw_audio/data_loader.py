# -*- coding: utf-8 -*-

import os

import numpy as np
import soundfile as sf

N = 16384


def read_file(path):
    sound, sr = sf.read(path)
    return sound[:N].reshape(N, 1).astype('float32')


def get_X_Y(data, some=lambda x: x[:]):
    # create duplicates for clean data as many as we have data in ds
    X = np.array(some(data['clean']) * len(data))

    Y = np.empty(shape=(0, N, 1))
    for group in data:
        d = data[group]
        y = np.array(some(d))
        Y = np.vstack((Y, y))

    return X, np.array(Y)


def load_train_test():
    data = {}

    base = "audio"
    spectrums_path = os.path.join('data', base)
    for root, dirs, files in os.walk(spectrums_path):
        if len(files) == 0:
            continue
        group = root.split(os.sep)[-1]
        if group == base:
            continue
        data[group] = []
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                data[group].append(read_file(file_path))

    X_train, Y_train = get_X_Y(data, lambda x: x[:20])
    X_test, Y_test = get_X_Y(data, lambda x: x[20:])

    return X_train, X_test, Y_train, Y_test


if __name__ == '__main__':
    load_train_test()