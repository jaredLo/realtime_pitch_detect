import numpy as np
import sounddevice as sd
from scipy.signal import fftconvolve
from numpy import argmax, diff

fs = 44100  # sample rate
duration = 5  # duration of the recording in seconds


def freq_from_autocorr(sig, fs):
    # Calculate autocorrelation and discard negative lags
    corr = fftconvolve(sig, sig[::-1], mode="full")
    corr = corr[len(corr) // 2 :]

    # Find the first low point
    d = diff(corr)
    nonzero_points = np.nonzero(d > 0)

    if nonzero_points[0].size > 0:
        start = nonzero_points[0][0]
        # Find the next peak after the low point
        peak = argmax(corr[start:]) + start

        if 1 < peak < len(corr) - 1:  # Check if peak is not at a boundary
            px, py = parabolic(corr, peak)
            return fs / px
        else:
            return 0  # Return 0 if peak is at a boundary
    else:
        return 0  # Return 0 if no nonzero derivative points found


def parabolic(f, x):
    """Quadratic interpolation for estimating the true position of an
    inter-sample maximum when nearby samples are known."""
    xv = 1 / 2.0 * (f[x - 1] - f[x + 1]) / (f[x - 1] - 2 * f[x] + f[x + 1]) + x
    yv = f[x] - 1 / 4.0 * (f[x - 1] - f[x + 1]) * (xv - x)
    return (xv, yv)


def callback(indata, frames, time, status):
    volume_norm = np.linalg.norm(indata) * 10
    if volume_norm > 0.01:  # adjust this value to change the volume threshold
        freq = freq_from_autocorr(indata[:, 0], fs)
        if 80 < freq < 1320:  # only consider pitches within the guitar range
            print("Estimated frequency: {:.2f} Hz".format(freq))


try:
    with sd.InputStream(device=0, callback=callback):
        print("Press Ctrl+C to stop")
        while True:
            pass
except KeyboardInterrupt:
    print("Stopped")
