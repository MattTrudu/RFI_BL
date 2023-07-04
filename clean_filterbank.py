import numpy as np
import matplotlib.pyplot as plt
from sigpyproc.Filterbank import Filterbank
from sigpyproc.Readers import FilReader
from scipy.signal import correlate
from scipy.linalg import toeplitz
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter
import os
import sys
import argparse

def elements_for_10_percent_sum(array):
    total_sum = np.sum(array)
    target_sum = 0.1 * total_sum  # 10% of the total sum
    sorted_indices = np.argsort(array)  # Sort the array indices in ascending order
    cumulative_sum = np.cumsum(array[sorted_indices])  # Calculate cumulative sum
    num_elements = np.searchsorted(cumulative_sum, target_sum, side='right') + 1
    return num_elements

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

def read_and_clean(filename, output_dir = os.getcwd(),  output_name = None ):

    """
    filename : name of the filterbankfile
    outputname : name of the filterbank cleaned
    gulp: number of time bins to read
    """

    filedir, name = os.path.split(filename)

    if output_name is None:
        output_name = name.replace(".fil","") + "_cleaned" + ".fil"


    filterbank = FilReader(filename)

    nsamp = filterbank.header.nsamples
    nchan = filterbank.header.nchans
    nbits = filterbank.header.nbits
    df    = filterbank.header.foff
    dt    = filterbank.header.tsamp
    print(nbits)

    outfile = filterbank.header.prepOutfile(os.path.join(output_dir,output_name), back_compatible = True, nbits = nbits)
    channels = np.arange(0, nchan)

    data = filterbank.readBlock(0, nsamp) # (nchans, nsamp)

    if nbits == 8:
        datawrite = data.T.astype("uint8")
    if nbits == 16:
        datawrite = data.T.astype("uint16")
    if nbits == 32:
        datawrite = data.T.astype("uint32")
    else:
        raise ValueError("Invalid value for nbits. Supported values are 8, 16, and 32.")

    outfile.cwrite(datawrite.ravel())

    outfile.close()

def _get_parser():
    """
    Argument parser.
    """

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Clean a SIGPROC filterbank file from RFI and produces a cleaned filterbank. It works only with > 8-bits filterbanks...")
    parser.add_argument('-f',
                        '--fil_file',
                        action = "store" ,
                        help = "SIGPROC .fil file to be processed (REQUIRED)",
                        required = True)
    parser.add_argument('-o',
                        '--output_dir',
                        action = "store" ,
                        help = "Output directory (Default: your current path)",
                        default = "%s/"%(os.getcwd())
                        )
    parser.add_argument('-n',
                        '--output_name',
                        action = "store" ,
                        help = "Output File Name (Default: filename_cleaned.fil)",
                        default = None
                        )


    return parser.parse_args()


if __name__ == '__main__':

    args = _get_parser()

    filename    = args.fil_file
    output_dir  = args.output_dir
    output_name = args.output_name


    read_and_clean(filename,
                output_dir  = output_dir,
                output_name = output_name,
                )



    """

    tstart = 108.49
    nstart = np.rint(tstart / dt).astype(np.int)
    twin = 60*1e-3
    nwin =  np.rint(twin / dt).astype(np.int)
    plt.figure()
    plt.imshow(data[:,nstart : nstart + nwin], aspect = "auto")
    plt.xlabel("Time Bins")
    plt.ylabel("Frequency Channels")
    plt.savefig("zoom.png")
    spectrum = data.mean(1)
    badchans = sk_filter(data.T, df, dt, sigma=3)

    timeseries = data.mean(0)
    mu    = timeseries.mean()
    sigma = timeseries.std()

    baseline = gaussian_filter(timeseries, 1, truncate = 1)
    badbins = (baseline < mu - 3 * sigma) | (baseline > mu + 3 * sigma)
    data[badchans,:] = 0
    data[:, badbins] = 0
    bins = np.arange(nsamp)

    plt.figure()
    plt.plot(timeseries)
    plt.plot(bins, baseline)
    plt.plot(bins[badbins], timeseries[badbins], "o")
    plt.savefig(f"ts.png")
    plt.figure()
    plt.plot(channels, spectrum)
    plt.plot(channels[badchans], spectrum[badchans], "o")
    plt.savefig(f"spec.png")



    nchunks = nsamp // gulp
    channels = np.arange(0, nchan)

    for ii in range(0, nsamp, gulp):
        dataproc = data[:, ii * gulp : (ii + 1) * gulp]
        badchans = sk_filter(dataproc.T, df, dt, sigma=4)
        print("Here")
        data[badchans, ii * gulp : (ii + 1) * gulp] = 0
        spectrum = dataproc.mean(1)
        plt.figure()
        plt.plot(channels, spectrum)
        plt.plot(channels[badchans], spectrum[badchans], "o")
        plt.savefig(f"{ii}.png")


    dataproc = data[:, nchunks * gulp : -1]
    badchans = sk_filter(dataproc.T, df, dt, sigma=4)
    data[badchans, nchunks * gulp : -1] = 0
    """
