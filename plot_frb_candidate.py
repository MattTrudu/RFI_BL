import numpy as np
import matplotlib.pyplot as plt
from sigpyproc.Filterbank import Filterbank
from sigpyproc.Readers import FilReader
from scipy.signal import correlate
from scipy.linalg import toeplitz
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import os
import sys
import argparse

def calc_N(channel_bandwidth, tsamp):

    tn = np.abs(1 / (channel_bandwidth * 10 ** 6))
    return np.round(tsamp / tn)

def spectral_kurtosis(data, N=1, d=None):

    zero_mask = data == 0
    data = np.ma.array(data.astype(float), mask=zero_mask)
    S1 = data.sum(0)
    S2 = (data ** 2).sum(0)
    M = data.shape[0]
    if d is None:
        d = (np.nanmean(data.ravel()) / np.nanstd(data)) ** 2
    return ((M * d * N) + 1) * ((M * S2 / (S1 ** 2)) - 1) / (M - 1)

def median_abs_deviation(sk_c):
    median = np.median(sk_c)
    mad = np.median(np.abs(sk_c - median))
    return mad

def sk_filter(data, channel_bandwidth, tsamp, N=None, d=None, sigma=5):

    if not N:
        N = calc_N(channel_bandwidth, tsamp)
    sk = spectral_kurtosis(data, d=d, N=N)
    nan_mask = np.isnan(sk)
    sk[nan_mask] = np.nan
    sk_c = sk[~nan_mask]
    std = 1.4826 * median_abs_deviation(sk_c)
    h = np.median(sk_c) + sigma * std
    l = np.median(sk_c) - sigma * std
    mask = (sk < h) & (sk > l)
    bad_channels = ~mask
    return bad_channels

def dispersion_delay(fstart, fstop, dms = None):

    """
    Simply computes the delay (in seconds!) due to dispersion. It accepts either a single DM value or a list of DMs. The output will be, respectively, a scalar or a list of delays.
    """

    return (
        4148808.0
        * dms
        * (1 / fstart ** 2 - 1 / fstop ** 2)
        / 1000
    )

def dedisperse(wfall, DM, freq, dt, ref_freq="bottom"):
    """
    Dedisperse a wfall matrix to DM.
    """

    k_DM = 1. / 2.41e-4
    dedisp = np.zeros_like(wfall)

    # pick reference frequency for dedispersion
    if ref_freq == "top":
        reference_frequency = freq[-1]
    elif ref_freq == "center":
        center_idx = len(freq) // 2
        reference_frequency = freq[center_idx]
    elif ref_freq == "bottom":
        reference_frequency = freq[0]
    else:
        #print "`ref_freq` not recognized, using 'top'"
        reference_frequency = freq[-1]

    shift = (k_DM * DM * (reference_frequency**-2 - freq**-2) / dt).round().astype(int)
    for i,ts in enumerate(wfall):
        dedisp[i] = np.roll(ts, shift[i])
    return dedisp

def plot_candidate(filename,
    tcand = 0,
    dmcand = 0,
    output_dir = os.getcwd(),
    output_name = "candidate",
    format_file = ".png",
    save_flag=True,
    sk_flag = False,
    sk_sig = 3
    ):

    filedir, name = os.path.split(filename)

    filterbank = FilReader(filename)

    nsamp = filterbank.header.nsamples
    nchan = filterbank.header.nchans
    nbits = filterbank.header.nbits
    df = filterbank.header.foff
    dt = filterbank.header.tsamp
    ftop = filterbank.header.ftop
    fbot = filterbank.header.fbottom
    fc = filterbank.header.fcenter

    channels = np.arange(0, nchan, 1)

    freqs = np.linspace(ftop, fbot, nchan)
    time = np.linspace(0, nsamp * dt, nsamp)

    delay = dispersion_delay(fbot, ftop, dms = dmcand)

    ncand  = np.rint(tcand / dt).astype("int")
    ndelay = np.rint(delay / dt).astype("int")

    data = filterbank.readBlock(ncand, ndelay)

    print(data.shape)

    if (sk_flag is True):
        badchans = sk_filter(data.T, df, dt, sigma = sk_sig)
        data[badchans, :] = 0

    figure = plt.figure(figsize = (10,7))
    size = 12

    widths0  = [0.8,0.2]
    widths1  = [1]
    heights0 = [0.2,0.4,0.4]
    heights1 = [0.2,0.4,0.4]


    gs0  = plt.GridSpec(3,2,hspace = 0.0 , wspace = 0,  width_ratios = widths0, height_ratios = heights0, top = 0.99 , bottom = 0.1, right = 0.55, left = 0.10)
    gs1  = plt.GridSpec(3,1,hspace = 0.0 , wspace = 0,  width_ratios = widths1, height_ratios = heights1, top = 0.99 , bottom = 0.1, right = 0.99, left = 0.65)

    ax0_00 = plt.subplot(gs0[0,0])
    #ax0_01 = plt.subplot(gs0[0,1])
    ax0_10 = plt.subplot(gs0[1,0])
    ax0_11 = plt.subplot(gs0[1,1])
    ax0_20 = plt.subplot(gs0[2,0])
    ax0_21 = plt.subplot(gs0[2,1])

    #ax1_00 = plt.subplot(gs1[0,0])
    #ax1_10 = plt.subplot(gs1[1,0])
    ax1_20 = plt.subplot(gs1[2,0])

    size = 15
    ax0_00.set_xticks([])
    ax0_00.set_yticks([])
    ax0_10.set_xticks([])
    ax0_11.set_xticks([])
    ax0_11.set_yticks([])
    ax0_21.set_xticks([])
    ax0_21.set_yticks([])
    ax0_00.margins(x=0)
    ax0_11.margins(y=0)

    ax0_10.set_ylabel("Frequency (MHz)", size = size)
    ax0_20.set_ylabel(r"DM (pc$\times$cm$^{-3}$)", size = size)
    ax0_20.set_xlabel("Time (ms)", size = size)

    ax1_20.set_ylabel("Frequency (MHz)", size = size)
    ax1_20.set_xlabel("Time (s)", size = size)

    ax1_20.imshow(data, aspect = "auto", extent = (0, delay, freqs[-1], freqs[0]), cmap = "inferno")

    if save_flag:
        output_name = f"{output_name}.{format_file}"
        plt.savefig(os.path.join(output_dir, output_name))
    else:
        plt.show()



def _get_parser():
    """
    Argument parser.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Read a SIGPROC filterbank file and plot an FRB candidate.",
    )
    parser.add_argument(
        "-f",
        "--fil_file",
        action = "store",
        help = "SIGPROC .fil file to be processed (REQUIRED).",
        required=True,
    )
    parser.add_argument(
        "-t",
        "--time_cand",
        type = float,
        help = "Arrival time of the candidate in seconds.",
        required=True,
    )
    parser.add_argument(
        "-d",
        "--dm_cand",
        type = float,
        help = "Dispersion measure of the candidate in pc cm^-3.",
        default = 0,
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        action="store",
        help="Output directory (Default: your current path).",
        default="%s/" % (os.getcwd()),
    )
    parser.add_argument(
        "-n",
        "--output_name",
        action="store",
        help="Output File Name (Default: data.png).",
        default=None,
    )
    parser.add_argument(
        "-s",
        "--save_data",
        help="Save the candidate plot.",
        action="store_true",
    )
    parser.add_argument(
        "-ff",
        "--file_format",
        action="store",
        help="Format of the candidate image (Default: .png).",
        default=".png",
    )

    return parser.parse_args()

if __name__ == "__main__":

    args = _get_parser()

    filename    = args.fil_file
    output_dir  = args.output_dir
    output_name = args.output_name
    save_flag   = args.save_data
    fileformat  = args.file_format
    tcand       = args.time_cand
    dmcand      = args.dm_cand

    plot_candidate(filename,
        tcand = tcand,
        dmcand = dmcand,
        output_dir = os.getcwd(),
        output_name = "candidate",
        format_file = ".png",
        save_flag=True
        )
