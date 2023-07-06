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
    save_flag=True
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

    ncand  = int(tcand // dt)
    ndelay = int(tcand // dt)

    data = filterbank.readBlock(ncand, ncand + ndelay)

    print(data.shape)

    plt.figure(figsize = (10,5))

    plt.imshow(data, aspect = "auto")
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
        action="store",
        help="SIGPROC .fil file to be processed (REQUIRED).",
        required=True,
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

    filename = args.fil_file
    output_dir = args.output_dir
    output_name = args.output_name
    save_flag = args.save_data
    fileformat = args.file_format

    plot_candidate(filename,
        tcand = 0,
        dmcand = 0,
        output_dir = os.getcwd(),
        output_name = "candidate",
        format_file = ".png",
        save_flag=True
        )
