# -*- coding: utf-8 -*-

import numpy as np
from scipy import misc


def load_train_test(path="data/spectrums/sp{0:02d}.png"):
    files = range(1, 31)

    imgs = [misc.imread(path.format(i))[:128, :128, 0].reshape((128, 128, 1)).astype('float32') / 255. for i in files]
    return np.array(imgs[:20]), np.array(imgs[20:]), np.array(imgs[:20]), np.array(imgs[20:])


def load_test(files):
    imgs = [misc.imread(file)[:128, :128, 0].reshape((128, 128, 1)).astype('float32') / 255. for file in files]
    return np.array(imgs)
