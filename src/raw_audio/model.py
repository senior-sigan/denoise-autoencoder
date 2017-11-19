# -*- coding: utf-8 -*-
from keras import Input, metrics
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.engine import Model
from keras.layers import Conv1D, MaxPooling1D, UpSampling1D, K
from keras.optimizers import Adadelta, SGD

NAME = "RawAudioCDNNAutoEncoder_7"


def build():
    inputs = Input(shape=(16384, 1))

    # Encoder
    conv = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(inputs)  # 16384x16
    conv = MaxPooling1D(pool_size=2, padding='same')(conv)  # 8192x64

    conv = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(conv)  # 8192x128
    conv = MaxPooling1D(pool_size=2, padding='same')(conv)  # 4096x128

    # conv = Conv1D(filters=256, kernel_size=3, padding='same', activation='relu')(conv)  # 4096x256
    # conv = MaxPooling1D(pool_size=2, padding='same')(conv)  # 2048x256
    #
    # # Decoder
    # conv = Conv1D(filters=256, kernel_size=3, padding='same', activation='relu')(conv)  # 2048x64
    # conv = UpSampling1D(size=2)(conv)  # 4096x256

    conv = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(conv)  # 4096x32
    conv = UpSampling1D(size=2)(conv)  # 8192x128

    conv = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(conv)  # 8192x16
    conv = UpSampling1D(size=2)(conv)  # 16384x64

    conv = Conv1D(filters=1, kernel_size=3, padding='same', activation='sigmoid')(conv)  # 16384x1

    autoencoder = Model(inputs, conv, name=NAME)
    return autoencoder


def train(model, x_train, y_train, x_test, y_test, epochs=100, batch_size=128):
    model.summary()
    model.compile(optimizer=Adadelta(lr=1.0, decay=0.2),
                  loss=K.binary_crossentropy,
                  metrics=[metrics.binary_accuracy, metrics.mean_squared_error])

    model.fit(x=x_train, y=y_train,
              epochs=epochs,
              batch_size=batch_size,
              shuffle=True,
              validation_data=(x_test, y_test),
              callbacks=[TensorBoard(
                  log_dir="/tmp/tensorflow/{}".format(NAME),
                  write_images=True,
                  histogram_freq=5,
                  batch_size=batch_size
              ), ModelCheckpoint(
                  "models/" + NAME + "/weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5",
                  monitor='val_loss',
                  verbose=1,
                  save_best_only=True,
                  mode='auto'
              )])
