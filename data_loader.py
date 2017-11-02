# -*- coding: utf-8 -*-

import numpy as np
from scipy import misc
from sklearn.model_selection import train_test_split


def load_data():
    files = range(1, 31)
    path = "data/spectrums/sp{0:02d}.png"

    imgs = [misc.imread(path.format(i))[:128, :128, 0].reshape((128, 128, 1)).astype('float32') / 255. for i in files]
    return [np.array(t) for t in train_test_split(imgs, imgs, test_size=0.33)]
