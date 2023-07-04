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

def eigenbasis(matrix):

    """
    Compute the eigenvalues and the eigenvectors of a square matrix and return the eigenspectrum (eigenvalues sorted in decreasing order) and the sorted eigenvectors respect
    to the eigenspectrum for the KLT analysis
    """

    eigenvalues,eigenvectors = np.linalg.eigh(matrix)

    if eigenvalues[0] < eigenvalues[-1]:
        eigenvalues = np.flipud(eigenvalues)
        eigenvectors = np.fliplr(eigenvectors)
    eigenspectrum = eigenvalues
    return eigenspectrum,eigenvectors

def count_elements_for_threshold(arr, threshold):
    sorted_arr = np.sort(arr)[::-1]  # Sort array in descending order
    total_sum = np.sum(sorted_arr)
    cumulative_sum = np.cumsum(sorted_arr)
    num_elements = np.searchsorted(cumulative_sum, threshold * total_sum, side='right') + 1
    return num_elements

def klt(signals, threshold):

    R = np.cov((signals-np.mean(signals,axis=0)),rowvar=False)

    eigenspectrum,eigenvectors = eigenbasis(R)

    neig = count_elements_for_threshold(eigenspectrum, threshold)

    coeff = np.matmul((signals[:,:]-np.mean(signals,axis=0)),np.conjugate((eigenvectors[:,:])))
    recsignals = np.matmul(coeff[:,0:int(neig)],np.transpose(eigenvectors[:,0:int(neig)])) + np.mean(signals,axis=0)

    return neig,eigenspectrum,eigenvectors,recsignals


def read_and_clean(filename,
                   output_dir = os.getcwd(),
                   output_name = None,
                   sk_flag = False,
                   sk_sig = 3,
                   klt_clean = False,
                   var_frac = 0.3,
                   klt_window = 1024):



    filedir, name = os.path.split(filename)

    if output_name is None:
        output_name = name.replace(".fil","") + "_cleaned" + ".fil"


    filterbank = FilReader(filename)

    nsamp = filterbank.header.nsamples
    nchan = filterbank.header.nchans
    nbits = filterbank.header.nbits
    df    = filterbank.header.foff
    dt    = filterbank.header.tsamp


    outfile = filterbank.header.prepOutfile(os.path.join(output_dir,output_name), back_compatible = True, nbits = nbits)
    channels = np.arange(0, nchan)

    data = filterbank.readBlock(0, nsamp) # (nchans, nsamp)

    if (sk_flag is True):
        badchans = sk_filter(data.T, df, dt, sigma = sk_sig)
        data[badchans, :] = 0

    if klt_clean:
    nchunks = nsamp // klt_window
    remainder = nsamp % klt_window

    for ii in tqdm(range(nchunks + 1)):
        start = ii * klt_window
        end = start + klt_window

        if ii == nchunks:
            end = nsamp

        datagrabbed = data[:, start:end]
        neig, ev, evecs, rfitemplate = klt(datagrabbed, var_frac)
        data[:, start:end] -= rfitemplate    

    """
    if (klt_clean is True):
        nchunks = nsamp // klt_window
        for ii in tqdm(range(nchunks)):
            datagrabbed = data[:,ii * klt_window : (ii + 1) * klt_window]
            neig, ev, evecs, rfitemplate = klt(datagrabbed, var_frac)
            data[:,ii * klt_window : (ii + 1) * klt_window] = data[:,ii * klt_window : (ii + 1) * klt_window] - rfitemplate
        datagrabbed = data[:,nchunks * klt_window : -1]
        neig, ev, evecs, rfitemplate = klt(datagrabbed, var_frac)
        data[:,nchunks * klt_window : -1] = data[:,nchunks * klt_window : -1] - rfitemplate
    """
    if int(nbits) == int(8):
        datawrite = data.T.astype("uint8")
    if int(nbits) == int(16):
        datawrite = data.T.astype("uint16")
    if int(nbits) == int(32):
        datawrite = data.T.astype("uint32")


    outfile.cwrite(datawrite.ravel())

    outfile.close()

def _get_parser():
    """
    Argument parser.
    """

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description = "Clean a SIGPROC filterbank file from RFI and produces a cleaned filterbank" + "\n"
                      "It performs an RFI excision in frequency via spectral kurtosis " + "\n"
                      "It performs an RFI excision in time via a Gaussian thresholding" + "\n"
                      "It also makes an RFI template, computed via a PCA (KLT), which will be subtracted to the data" + "\n"
                      "It works only with > 8-bits filterbanks...")
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
    parser.add_argument('-sk',
                        '--spectral_kurtosis',
                        help = "Find the bad channels via a spectral kurtosis (Bad channels will be set to zero). Default = False.",
                        action = 'store_true',
                        )
    parser.add_argument('-sksig',
                        '--spectral_kurtosis_sigma',
                        type = int,
                        default = 3,
                        action = "store" ,
                        help = "Sigma for the Spectral Kurtosis (Default: 3)"
                        )
    parser.add_argument('-klt',
                        '--karhunen_loeve_cleaning',
                        help = "Evaluate an RFI template via a KLT and remove it from the data. Default = False.",
                        action = 'store_true',
                        )
    parser.add_argument('-var_frac',
                        '--variance_fraction',
                        type = float,
                        default = 0.3,
                        action = "store" ,
                        help = "The fraction of the total variance of the signal to consider (between 0 and 1). The number of associated eigenvalues will be computed from this. (Default: 0.3)"
                        )
    parser.add_argument('-klt_win',
                        '--klt_window',
                        type = int,
                        default = 1024,
                        action = "store" ,
                        help = "Number of time bins to consider in each read to make the KLT. (Default: 1024)"
                        )

    return parser.parse_args()


if __name__ == '__main__':

    args = _get_parser()

    filename    = args.fil_file
    output_dir  = args.output_dir
    output_name = args.output_name
    sk_flag     = args.spectral_kurtosis
    sk_sig      = args.spectral_kurtosis_sigma
    klt_clean   = args.karhunen_loeve_cleaning
    var_frac    = args.variance_fraction
    kltwindow  = args.klt_window


    read_and_clean(filename,
                output_dir  = output_dir,
                output_name = output_name,
                sk_flag = sk_flag,
                sk_sig = sk_sig,
                klt_clean = klt_clean,
                var_frac = var_frac,
                klt_window = kltwindow
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
