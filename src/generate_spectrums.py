# -*- coding: utf-8 -*-

import numpy as np
import soundfile as sf
from matplotlib import pyplot as plt
from scipy import signal
import os

sound_len = 16384


def make_spectrum(sound, sample_rate):
    sound = sound[:sound_len]
    freqs, bins, Sxx = signal.stft(sound, nfft=256, fs=sample_rate, noverlap=128)
    Pxx = 10 * np.log10(np.abs(Sxx) + 0.000000000000001)  # dummy hack to avoid zero division
    return Pxx


def save_spectrum(name, image_path, sound_path):
    sound, sr = sf.read(sound_path.format(name))
    print("{}: {}".format(name, sound.shape))
    Pxx = make_spectrum(sound, sr)
    plt.imsave(image_path.format(name), Pxx, cmap='gray')


def main():
    # sound_path = 'data/clean/{}.wav'
    # image_path = 'data/spectrums/{}.png'

    sound_path = 'data/audio/car_10dB/{}.wav'
    image_path = 'data/test_spectrums/{}.png'

    for name in os.listdir('data/audio/car_10dB'):
        save_spectrum(name.replace(".wav", ""), image_path, sound_path)


if __name__ == '__main__':
    main()
