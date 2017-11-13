# -*- coding: utf-8 -*-

from encoder_clean_to_clean import build_model, train

from spectrums.data_loader import load_train_test


def main():
    X_train, X_test, y_train, y_test = load_train_test()
    model = build_model()
    train(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=20)

    model.save("models/final.h5")


if __name__ == '__main__':
    main()
