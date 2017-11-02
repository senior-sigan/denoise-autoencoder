# -*- coding: utf-8 -*-

from keras import backend as K
from keras import metrics
from keras.callbacks import TensorBoard
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.core import Activation
from keras.models import Model
from keras.optimizers import Adam
from matplotlib import pyplot as plt

from data_loader import load_data


def conv_layer(x, size, dropout):
    conv = Conv2D(size, (3, 3), padding='same')(x)
    conv = Activation('relu')(conv)
    # conv = Conv2D(size, (3, 3), padding='same')(conv)
    # conv = Activation('relu')(conv)
    # conv = Dropout(dropout)(conv)
    return conv


def build_model():
    inputs = Input(shape=(128, 128, 1))
    dropout = 0.1

    # 129x129x32
    conv = conv_layer(inputs, 32, dropout)

    # 64x64x32
    pool = MaxPooling2D(pool_size=(2, 2))(conv)

    # 64x64x64
    conv = conv_layer(pool, 64, dropout)

    # 32x32x64
    pool = MaxPooling2D(pool_size=(2, 2))(conv)

    # 32x32x128
    conv = conv_layer(pool, 128, dropout)

    # 16x16x128
    pool = MaxPooling2D(pool_size=(2, 2))(conv)

    # ---

    # 16x16x128
    conv = conv_layer(pool, 128, dropout)

    # 32x32x128
    up = UpSampling2D((2, 2))(conv)

    # 32x32x64
    conv = conv_layer(up, 64, dropout)

    # 64x64x64
    up = UpSampling2D((2, 2))(conv)

    # 64x64x32
    conv = conv_layer(up, 32, dropout)

    # 128x128x32
    up = UpSampling2D((2, 2))(conv)

    conv = Conv2D(1, (1, 1))(up)
    conv = Activation('sigmoid')(conv)

    model = Model(inputs, conv, name="autoencoder_sound")
    return model


def train(model, x_train, y_train, x_test, y_test, epochs=50, batch_size=128):
    model.summary()
    model.compile(optimizer=Adam(0.001),
                  loss=K.binary_crossentropy,
                  metrics=[metrics.binary_accuracy])

    model.fit(x=x_train, y=y_train,
              epochs=epochs,
              batch_size=batch_size,
              shuffle=True,
              validation_data=(x_test, y_test),
              callbacks=[TensorBoard(
                  log_dir="/tmp/tensorflow/autoencoder_sound",
                  write_images=True,
                  histogram_freq=5,
                  batch_size=batch_size
              )])


def save_predictions(prediction):
    n = len(prediction)
    for i in range(n):
        plt.imsave("out/sp{0:02d}.png".format(i + 1),
                   prediction[i].reshape(128, 128),
                   cmap='gray')


def main():
    X_train, X_test, y_train, y_test = load_data()
    model = build_model()
    train(model, X_train, y_train, X_test, y_test, epochs=300, batch_size=5)

    model.save("models/autoencoder.h5")

    save_predictions(model.predict(X_test))


if __name__ == '__main__':
    main()
