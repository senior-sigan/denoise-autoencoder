# -*- coding: utf-8 -*-

from data_loader import load_train_test
from matplotlib import pyplot as plt

from encoder_clean_to_clean import build_model, train


def save_predictions(prediction):
    n = len(prediction)
    for i in range(n):
        plt.imsave("out/sp{0:02d}.png".format(i + 21),
                   prediction[i].reshape(128, 128),
                   cmap='gray')


def main():
    X_train, X_test, y_train, y_test = load_train_test()
    model = build_model()
    train(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=5)

    model.save("models/final.h5")

    save_predictions(model.predict(X_test))


if __name__ == '__main__':
    main()