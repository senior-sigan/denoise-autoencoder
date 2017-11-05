# -*- coding: utf-8 -*-

import os

import numpy as np
from scipy import misc


def read_file(path):
    return misc.imread(path)[:128, :128, 0].reshape((128, 128, 1)).astype('float32') / 255.


def get_X_Y(data, some=lambda x: x[:]):
    # create duplicates for clean data as many as we have data in ds
    X = np.array(some(data['clean']) * len(data))

    Y = np.empty(shape=(0, 128, 128, 1))
    for group in data:
        y = np.array(some(data[group]))
        Y = np.vstack((Y, y))

    return X, np.array(Y)


def load_train_test():
    data = {}

    spectrums_path = os.path.join('data', 'spectrums')
    for root, dirs, files in os.walk(spectrums_path):
        if len(files) == 0:
            continue
        group = root.split(os.sep)[-1]
        data[group] = []
        for file in files:
            file_path = os.path.join(root, file)
            data[group].append(read_file(file_path))

    X_train, Y_train = get_X_Y(data, lambda x: x[:20])
    X_test, Y_test = get_X_Y(data, lambda x: x[20:])

    return X_train, X_test, Y_train, Y_test


def load_test(files):
    pass


if __name__ == '__main__':
    for i in load_train_test():
        print(i.shape)
