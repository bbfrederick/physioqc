"""Denoising metrics for chest belt recordings."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from peakdet import Physio, operations
from scipy.signal import resample, welch
from scipy.stats import pearsonr

from .. import references
from ..due import due
from .utils import hamming, physio_or_numpy


def envelopedetect(data, upperpass=0.05, order=5):
    """

    Parameters
    ----------
    data
    upperpass: float
        Upper pass cutoff frequency in Hz
    order: int
        Butterworth filter order
    Returns
    -------
    einferior: ndarray
        The lower edge of the signal envelope
    esuperior: ndarray
        The upper edge of the signal envelope

    """
    npdata = data.data
    targetfs = data.fs
    esuperior = (
        2.0
        * operations.filter_physio(
            Physio(np.square(np.where(npdata > 0.0, npdata, 0.0)), fs=targetfs),
            [upperpass],
            "lowpass",
            order=order,
        ).data
    )
    esuperior = np.sqrt(np.where(esuperior > 0.0, esuperior, 0.0))
    einferior = (
        2.0
        * operations.filter_physio(
            Physio(np.square(np.where(npdata < 0.0, npdata, 0.0)), fs=targetfs),
            [upperpass],
            "lowpass",
            order=order,
        ).data
    )
    einferior = -np.sqrt(np.where(einferior > 0.0, einferior, 0.0))
    return einferior, esuperior


@due.dcite(references.ROMANO_2023)
def respiratorysqi(
    rawresp, Fs, envelopelpffreq=0.075, slidingfilterpctwidth=10.0, debug=False
):
    """
    Implementation of Romano's method from A Signal Quality Index for Improving the Estimation of
    Breath-by-Breath Respiratory Rate During Sport and Exercise,
    IEEE SENSORS JOURNAL, VOL. 23, NO. 24, 15 DECEMBER 2023

    NB: In part A, Romano does not specify the upper pass frequency for the respiratory envelope filter.
        0.075Hz looks pretty good.
        In part B, the width of the sliding window bandpass filter is not specified.  We use a range of +/- 5% of the
        center frequency.

    Parameters
    ----------
    rawresp: ndarray
        The raw respiration signal
    Fs: float
        The sampling frequency of the respiration signal in Hz
    envelopelpffreq: float
        Cutoff frequency of the RMS envelope filter in Hz
    slidingfilterpctwidth: float
        Width of the sliding window bandpass filter in percent of the center frequency
    debug: bool
        Print debug information and plot intermediate results

    Returns
    -------
    breathlist: list
        List of "breathinfo" dictionaries for each detected breath.  Each breathinfo dictionary contains:
            "startime", "centertime", and "endtime" of each detected breath in seconds from the start of the waveform,
            and "correlation" - the Pierson correlation of that breath waveform with the average waveform.
    """
    # rawresp = physio_or_numpy(rawresp)

    # get the sample frequency down to around 25 Hz for respiratory waveforms
    targetfs = 25.0
    rawresp = Physio(rawresp, fs=Fs)
    rawresp = operations.interpolate_physio(rawresp, target_fs=targetfs)

    # A. Signal Preprocessing
    # Apply third order Butterworth bandpass, 0.01-2Hz
    prefiltered = operations.filter_physio(
        rawresp,
        [0.01, 2.0],
        "bandpass",
        order=3,
    )
    if debug:
        plt.plot(rawresp.data)
        plt.plot(prefiltered.data)
        plt.title("Raw and prefiltered respiratory signal")
        plt.legend(["Raw", "Prefiltered"])
        plt.show()
    if debug:
        print("prefiltered: ", prefiltered.data)

    # calculate the derivative
    derivative = np.gradient(prefiltered.data, 1.0 / targetfs)

    # normalize the derivative to the range of ~-1 to 1
    derivmax = np.max(derivative)
    derivmin = np.min(derivative)
    derivrange = derivmax - derivmin
    if debug:
        print(f"{derivmax=}, {derivmin=}, {derivrange=}")
    normderiv = 2.0 * (derivative - derivmin) / derivrange - 1.0
    if debug:
        plt.plot(normderiv)
        plt.title("Normalized derivative of respiratory signal")
        plt.legend(["Normalized derivative"])
        plt.show()

    # amplitude correct by flattening the envelope function
    einferior, esuperior = envelopedetect(
        Physio(normderiv, fs=targetfs), upperpass=envelopelpffreq, order=3
    )
    if debug:
        plt.plot(normderiv)
        plt.plot(esuperior)
        plt.plot(einferior)
        plt.legend(["Normalized derivative", "esuperior", "einferior"])
        plt.show()
    rmsnormderiv = (normderiv - einferior) / (esuperior - einferior)
    if debug:
        plt.plot(rmsnormderiv)
        plt.title(
            "Normalized derivative of respiratory signal after envelope correction"
        )
        plt.show()

    # B. Detection of breaths in sliding window
    seglength = 12.0
    segsamples = int(seglength * targetfs)
    segstep = 2.0
    stepsamples = int(segstep * targetfs)
    totaltclength = len(rawresp)
    numsegs = int((totaltclength - segsamples) // stepsamples)
    if (totaltclength - segsamples) % segsamples != 0:
        numsegs += 1
    peakfreqs = np.zeros((numsegs), dtype=np.float64)
    respfilteredderivs = rmsnormderiv * 0.0
    respfilteredweights = rmsnormderiv * 0.0
    for i in range(numsegs):
        if i < numsegs - 1:
            segstart = i * stepsamples
            segend = segstart + segsamples
        else:
            segstart = len(rawresp) - segsamples
            segend = segstart + segsamples
        segment = rmsnormderiv[segstart:segend] + 0.0
        segment *= hamming(segsamples)
        segment -= np.mean(segment)
        thex, they = welch(segment, targetfs, nfft=4096)
        peakfreqs[i] = thex[np.argmax(they)]
        respfilterorder = 1
        lowerfac = 1.0 - slidingfilterpctwidth / 200.0
        upperfac = 1.0 + slidingfilterpctwidth / 200.0
        lowerpass = peakfreqs[i] * lowerfac
        upperpass = peakfreqs[i] * upperfac
        if debug:
            print(peakfreqs[i], lowerfac, lowerpass, upperfac, upperpass)
        filteredsegment = operations.filter_physio(
            Physio(segment, fs=targetfs),
            [upperpass],
            "lowpass",
            order=respfilterorder,
        ).data
        filteredsegment -= np.mean(filteredsegment)
        if i < numsegs - 1:
            respfilteredderivs[
                i * stepsamples : (i * stepsamples) + segsamples
            ] += filteredsegment
            respfilteredweights[i * stepsamples : (i * stepsamples) + segsamples] += 1.0
        else:
            respfilteredderivs[-segsamples:] += filteredsegment
            respfilteredweights[-segsamples:] += 1.0
    respfilteredderivs /= respfilteredweights
    respfilteredderivs /= np.std(respfilteredderivs)
    if debug:
        print(peakfreqs)
        plt.plot(rmsnormderiv)
        plt.plot(respfilteredderivs)
        plt.title("Normalized derivative before and after targeted bandpass filtering")
        plt.legend(["Normalized derivative", "Filtered normalized derivative"])
        plt.show()

    # C. Breaths segmentation
    # The fastest credible breathing rate is 20 breaths/min -> 3 seconds/breath, so set the dist to be 50%
    # of that in points
    thedist = int((3.0 * targetfs) / 2.0)
    peakinfo = operations.peakfind_physio(respfilteredderivs, thresh=0.05, dist=thedist)
    if debug:
        ax = operations.plot_physio(peakinfo)
        plt.show()
    thepeaks = peakinfo.peaks

    # D. Similarity Analysis and Exclusion of Unreliable Breaths
    numbreaths = len(thepeaks) - 1
    scaledpeaklength = 100
    thescaledbreaths = np.zeros((numbreaths, scaledpeaklength), dtype=np.float64)
    thebreathlocs = np.zeros((numbreaths), dtype=np.float64)
    thebreathcorrs = np.zeros((numbreaths), dtype=np.float64)
    breathlist = []
    for thisbreath in range(numbreaths):
        breathinfo = {}
        startpt = thepeaks[thisbreath]
        endpt = thepeaks[thisbreath + 1]
        thebreathlocs[thisbreath] = (endpt + startpt) / (targetfs * 2.0)
        breathinfo["starttime"] = startpt / targetfs
        breathinfo["endtime"] = endpt / targetfs
        breathinfo["centertime"] = thebreathlocs[thisbreath]
        thescaledbreaths[thisbreath, :] = resample(
            respfilteredderivs[startpt:endpt], scaledpeaklength
        )
        thescaledbreaths[thisbreath, :] -= np.min(thescaledbreaths[thisbreath, :])
        thescaledbreaths[thisbreath, :] /= np.max(thescaledbreaths[thisbreath, :])
        breathlist.append(breathinfo)
        if debug:
            plt.plot(thescaledbreaths[thisbreath, :], lw=0.1, color="#888888")
    averagebreath = np.mean(thescaledbreaths, axis=0)
    if debug:
        plt.plot(averagebreath, lw=2, color="black")
        plt.show()
    for thisbreath in range(numbreaths):
        thebreathcorrs[thisbreath] = pearsonr(
            averagebreath, thescaledbreaths[thisbreath, :]
        ).statistic
        breathlist[thisbreath]["correlation"] = thebreathcorrs[thisbreath]
    return breathlist


def plotbreathqualities(breathlist, totaltime=None):
    """
    Make an informational plot of quantifiability vs time for each detected breath

    Parameters
    ----------
    breathlist: list
        A list of breathinfo dictionaries output from respiratorysqi
    totaltime: float
        The maximum time to include in the plot.  If totaltime is None (default), plot every breath.

    Returns
    -------

    """
    # set up the color codes
    color_0p9 = "#888888"
    color_0p8 = "#aa6666"
    color_0p7 = "#cc4444"
    color_bad = "#ff0000"

    if totaltime is None:
        totaltime = breathlist[-1]["endtime"]

    # unpack the breath information
    numbreaths = len(breathlist)
    thebreathlocs = np.zeros((numbreaths), dtype=np.float64)
    thebreathcorrs = np.zeros((numbreaths), dtype=np.float64)
    for thisbreath in range(numbreaths):
        thebreathlocs[thisbreath] = breathlist[thisbreath]["centertime"]
        thebreathcorrs[thisbreath] = breathlist[thisbreath]["correlation"]

    # plot the breath correlations, with lines indicating thresholds
    plt.plot(thebreathlocs, thebreathcorrs, ls="-", marker="o")
    plt.hlines(
        [0.9, 0.8, 0.7, 0.6],
        0.0,
        totaltime,
        colors=[color_0p9, color_0p8, color_0p7, color_bad],
        linestyles="solid",
        label=["th=0.9", "th=0.8", "th=0.7", "th=0.6"],
    )
    plt.title("Quality evaluation for each breath")
    plt.xlabel("Time in seconds")
    plt.ylabel("Correlation with average breath")
    plt.xlim([0, totaltime])
    plt.ylim([0, 1.05])
    plt.show()


def plotbreathwaveformwithquality(waveform, breathlist, Fs, plottype="rectangle"):
    """
    Make an informational plot of the respiratory waveform with the quantifiability of each detected breath as a
    color overlay.

    Parameters
    ----------
    waveform: ndarray
        The respiratory waveform
    breathlist: list
        A list of breathinfo dictionaries output from respiratorysqi
    Fs: float
        The sampling frequency of the respiratory waveform in Hz
    plottype: str
        The type of plot to make.  If plottype is rectangle (default), overlay a colored rectangle to show the
        quantifiability of each detected breath.  If anything else, use the waveform line color to indicate the
        quantifiability.
    Returns
    -------

    """
    # now plot the respiratory waveform, color coded for quality

    # set up the color codes
    color_0p9 = "#ff000000"
    color_0p8 = "#ff000044"
    color_0p7 = "#ff000088"
    color_bad = "#ff0000aa"

    # unpack the breath information
    numbreaths = len(breathlist)
    thebreathlocs = np.zeros((numbreaths), dtype=np.float64)
    thebreathcorrs = np.zeros((numbreaths), dtype=np.float64)
    for thisbreath in range(numbreaths):
        thebreathlocs[thisbreath] = breathlist[thisbreath]["centertime"]
        thebreathcorrs[thisbreath] = breathlist[thisbreath]["correlation"]

    # initialize the plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # plot the whole line if we're doing rectangle occlusion
    if plottype == "rectangle":
        xvals = np.linspace(0.0, len(waveform) / Fs, len(waveform), endpoint=False)
        plt.plot(xvals, waveform, color="black")
    yrange = np.max(waveform) - np.min(waveform)
    ymax = np.min(waveform) + yrange * 1.05
    ymin = np.max(waveform) - yrange * 1.05
    for thisbreath in range(numbreaths):
        if thebreathcorrs[thisbreath] > 0.9:
            if plottype == "rectangle":
                thecolor = color_0p9
            else:
                thecolor = "black"
        elif thebreathcorrs[thisbreath] > 0.8:
            thecolor = color_0p8
        elif thebreathcorrs[thisbreath] > 0.7:
            thecolor = color_0p7
        else:
            thecolor = color_bad
        startpt = int(breathlist[thisbreath]["starttime"] * Fs)
        endpt = int(breathlist[thisbreath]["endtime"] * Fs)
        if plottype == "rectangle":
            therectangle = Polygon(
                (
                    (breathlist[thisbreath]["starttime"], ymin),
                    (breathlist[thisbreath]["starttime"], ymax),
                    (breathlist[thisbreath]["endtime"], ymax),
                    (breathlist[thisbreath]["endtime"], ymin),
                ),
                fc=thecolor,
                ec=thecolor,
                lw=0,
            )
            ax.add_patch(therectangle)
            pass
        else:
            if endpt == len(waveform) - 1:
                endpt -= 1
            xvals = np.linspace(
                startpt / Fs, (endpt + 1) / Fs, endpt - startpt + 1, endpoint=False
            )
            yvals = waveform[startpt : endpt + 1]
            plt.plot(xvals, yvals, color=thecolor)
    plt.ylim([ymin, ymax])
    plt.title("Respiratory waveform, color coded by quantifiability")
    plt.xlabel("Time in seconds")
    plt.ylabel("Amplitude (arbitrary units)")
    plt.xlim([0.0, len(waveform) / Fs])
    plt.show()


def summarizebreaths(breathlist):
    """
    Print summary information about each detected breath.

    Parameters
    ----------
    breathlist: list
        A list of breathinfo dictionaries output from respiratorysqi

    Returns
    -------

    """
    numbreaths = len(breathlist)
    for thisbreath in range(numbreaths):
        thebreathinfo = breathlist[thisbreath]
        print(
            f"{thisbreath},{thebreathinfo['starttime']},{thebreathinfo['endtime']},{thebreathinfo['centertime']},{thebreathinfo['correlation']}"
        )
