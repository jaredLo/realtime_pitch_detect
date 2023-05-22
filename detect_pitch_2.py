import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import fftconvolve
from numpy.fft import rfft, irfft
from numpy import argmax, mean, diff, log


def freq_from_autocorr(sig, fs):
    # Calculate autocorrelation and discard negative lags
    corr = fftconvolve(sig, sig[::-1], mode="full")
    corr = corr[len(corr) // 2 :]

    # Find the first low point
    d = diff(corr)
    start = np.nonzero(d > 0)[0][0]

    # Find the next peak after the low point
    peak = argmax(corr[start:]) + start
    px, py = parabolic(corr, peak)

    return fs / px


def parabolic(f, x):
    """Quadratic interpolation for estimating the true position of an
    inter-sample maximum when nearby samples are known."""
    xv = 1 / 2.0 * (f[x - 1] - f[x + 1]) / (f[x - 1] - 2 * f[x] + f[x + 1]) + x
    yv = f[x] - 1 / 4.0 * (f[x - 1] - f[x + 1]) * (xv - x)
    return (xv, yv)


filename = "e_2.wav"  # replace with your file name
fs, data = wav.read(filename)

# Convert to mono if audio is stereo
if len(data.shape) > 1:
    data = data.mean(axis=1)

data = data[20000:40000]  # select a slice
print("Estimated frequency: {:.2f} Hz".format(freq_from_autocorr(data, fs)))
