# -*- coding: utf-8 -*-

from keras import backend as K
from keras import metrics
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import Input, Conv2D
from keras.layers.core import Activation
from keras.models import Model
from keras.optimizers import Adadelta


def conv_layer(x, size, dropout, filter_size=(3, 3)):
    conv = Conv2D(size, filter_size, padding='same')(x)
    conv = Activation('relu')(conv)
    # conv = Conv2D(size, (3, 3), padding='same')(conv)
    # conv = Activation('relu')(conv)
    # conv = Dropout(dropout)(conv)
    return conv


def build_model():
    inputs = Input(shape=(128, 128, 1))
    dropout = 0.3
    filter_size = (3, 3)

    conv = conv_layer(inputs, 64, dropout, filter_size)

    conv = conv_layer(conv, 64, dropout, filter_size)

    conv = conv_layer(conv, 64, dropout, filter_size)

    conv = conv_layer(conv, 64, dropout, filter_size)

    conv = Conv2D(1, (1, 1))(conv)
    conv = Activation('sigmoid')(conv)

    model = Model(inputs, conv, name="autoencoder_sound")
    return model


def train(model, x_train, y_train, x_test, y_test, epochs=100, batch_size=128):
    model.summary()
    model.compile(optimizer=Adadelta(),
                  loss=K.binary_crossentropy,
                  metrics=[metrics.binary_accuracy, metrics.mean_squared_error])

    model.fit(x=x_train, y=y_train,
              epochs=epochs,
              batch_size=batch_size,
              shuffle=True,
              validation_data=(x_test, y_test),
              callbacks=[TensorBoard(
                  log_dir="/tmp/tensorflow/autoencoder_sound",
                  write_images=False,
                  histogram_freq=5,
                  batch_size=batch_size
              ), ModelCheckpoint(
                  "models/weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5",
                  monitor='val_loss',
                  verbose=1,
                  save_best_only=True,
                  mode='auto'
              )])
