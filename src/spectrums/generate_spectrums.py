# -*- coding: utf-8 -*-

import os

import numpy as np
import soundfile as sf
from matplotlib import pyplot as plt
from scipy import signal

sound_len = 16384


def make_spectrum(sound, sample_rate):
    sound = sound[:sound_len]
    freqs, bins, Sxx = signal.stft(sound, nfft=256, fs=sample_rate, noverlap=128)
    Pxx = 10 * np.log10(np.abs(Sxx) + 0.000000000000001)  # dummy hack to avoid zero division
    return Pxx


def save_spectrum(sound_path, image_path):
    name = os.path.split(sound_path)[-1]
    sound, sr = sf.read(sound_path)
    Pxx = make_spectrum(sound, sr)
    plt.imsave(os.path.join(image_path, name + ".png"), Pxx, cmap='gray')


def main():
    sound_path = os.path.join('data', 'audio')
    image_path = os.path.join('data', 'spectrums')

    for root, dirs, files in os.walk(sound_path):
        if len(files) == 0:
            continue
        group = root.split(os.sep)[-1]
        spectrum_path = os.path.join(image_path, group)
        os.makedirs(spectrum_path, exist_ok=True)
        files = [os.path.join(root, f) for f in files]
        for file in files:
            save_spectrum(file, spectrum_path)


if __name__ == '__main__':
    main()
