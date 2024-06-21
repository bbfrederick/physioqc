"""Denoising metrics for chest belt recordings."""

import numpy as np
from .utils import dobpfiltfilt, dolpfiltfilt
import matplotlib.pyplot as plt

from .. import references
from ..due import due


def respirationprefilter(rawresp, Fs, lowerpass=0.01, upperpass=2.0, order=1, debug=False):
    if debug:
        print(f"respirationprefilter: Fs={Fs} order={order}, lowerpass={lowerpass}, upperpass={upperpass}")
    return dobpfiltfilt(Fs, lowerpass, upperpass, rawresp, order, debug=debug)

def respenvelopefilter(squarevals, Fs, upperpass=0.1, order=8, debug=False):
    if debug:
        print(f"respenvelopefilter: Fs={Fs} order={order}, upperpass={upperpass}")
    return dolpfiltfilt(Fs, upperpass, squarevals, order, debug=debug)


def respiratorysqi(rawresp, Fs, debug=False):
    """Implementation of Romano's method from A Signal Quality Index for Improving the Estimation of
    Breath-by-Breath Respiratory Rate During Sport and Exercise,
    IEEE SENSORS JOURNAL, VOL. 23, NO. 24, 15 DECEMBER 2023"""

    # A. Signal Preprocessing
    # Apply first order Butterworth bandpass, 0.01-2Hz
    prefiltered = respirationprefilter(rawresp, Fs, debug=debug)
    if debug:
        plt.plot(rawresp)
        plt.plot(prefiltered)
        plt.show()
    if debug:
        print("prefiltered: ", prefiltered)

    # calculate the derivative
    derivative = np.gradient(prefiltered, 1.0 / Fs)
    if debug:
        plt.plot(prefiltered)
        plt.plot(derivative)
        plt.show()

    # normalize the derivative to the range of ~-1 to 1
    derivmax = np.max(derivative)
    derivmin = np.min(derivative)
    derivrange = derivmax - derivmin
    if debug:
        print(f"{derivmax=}, {derivmin=}, {derivrange=}")
    normderiv = 2.0 * (derivative - derivmin) / derivrange - 1.0
    if debug:
        plt.plot(normderiv)
        plt.show()

    # amplitude correct by flattening the envelope function
    esuperior = 2.0 * respenvelopefilter(np.square(np.where(normderiv > 0.0, normderiv, 0.0)), Fs)
    esuperior = np.sqrt(np.where(esuperior > 0.0, esuperior, 0.0))
    einferior = 2.0 * respenvelopefilter(np.square(np.where(normderiv < 0.0, normderiv, 0.0)), Fs)
    einferior = np.sqrt(np.where(einferior > 0.0, einferior, 0.0))
    if debug:
        plt.plot(normderiv)
        plt.plot(esuperior)
        plt.plot(-einferior)
        plt.show()
    rmsnormderiv = normderiv / (esuperior + einferior)
    if debug:
        plt.plot(rmsnormderiv)
        plt.show()

    # B. Detection of breaths in sliding window
    seglength = 12.0
    segsamples = seglength * Fs
    segstep = 2.0
    stepsamples = segstep * Fs
    totaltclength = len(rawresp)
    numsegs = int(totaltclength // segsamples)

    if totaltclength % segsamples != 0:
        numsegs += 1


    # C. Breaths segmentation

    # D. Similarity Analysis and Exclusion of Unreliable Breaths

    # E. Breath-by-Breath RR Assessment

    return normderiv


