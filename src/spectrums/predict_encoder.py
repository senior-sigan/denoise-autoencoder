# -*- coding: utf-8 -*-
import os

import numpy as np
from keras.models import load_model
from matplotlib import pyplot as plt

from spectrums.data_loader import read_file


def save_predictions(prediction):
    n = len(prediction)
    for i in range(n):
        plt.imsave("out_test/sp{0:02d}.png".format(i + 1),
                   prediction[i].reshape(128, 128),
                   cmap='gray')


def main():
    model = load_model('models/final.h5')

    path = 'data/spectrums/airport0dB'
    X_test = np.array([read_file(os.path.join(path, name)) for name in os.listdir(path)])
    save_predictions(model.predict(X_test))


if __name__ == '__main__':
    main()
