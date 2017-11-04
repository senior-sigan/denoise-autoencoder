# -*- coding: utf-8 -*-
import os

from keras.models import load_model
from matplotlib import pyplot as plt

from data_loader import load_test


def save_predictions(prediction):
    n = len(prediction)
    for i in range(n):
        plt.imsave("out_test/sp{0:02d}.png".format(i + 1),
                   prediction[i].reshape(128, 128),
                   cmap='gray')


def main():
    model = load_model('models/final.h5')

    files = ['data/test_spectrums/{}'.format(name) for name in os.listdir('data/test_spectrums')]
    X_test = load_test(files)
    save_predictions(model.predict(X_test))


if __name__ == '__main__':
    main()
