# -*- coding: utf-8 -*-

from raw_audio.data_loader import load_train_test
from raw_audio.model import build, train, NAME


def main():
    X_train, X_test, y_train, y_test = load_train_test()
    model = build()
    train(model, X_train, y_train, X_test, y_test, epochs=60, batch_size=14)

    model.save("models/{}/final.h5".format(NAME))


if __name__ == '__main__':
    main()
