import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sigpyproc.Filterbank import Filterbank
from sigpyproc.Readers import FilReader


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

def dedisperse(wfall, DM, freq, dt, ref_freq="top"):
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

def plot_data(filename,
                   output_dir = os.getcwd(),
                   output_name = "candidate",
                   grab_flag = False,
                   save_flag = True,
                   file_format = ".png",
                   tstart = 0,
                   tstop  = 1
                   ):


    filedir, name = os.path.split(filename)

    filterbank = FilReader(filename)

    nsamp = filterbank.header.nsamples
    nchan = filterbank.header.nchans
    nbits = filterbank.header.nbits
    df    = filterbank.header.foff
    dt    = filterbank.header.tsamp
    ftop  = filterbank.header.ftop
    fbot  = filterbank.header.fbot
    fc    = filterbank.header.fcenter

    channels = np.arange(0,nchan,1)

    freqs = np.linspace(ftop, fbot, nchan)
    time  = np.linspace(0, nsamp * dt)

    if (grab_flag is False):

        data = filterbank.readBlock(0, nsamp)
        timeseries = np.mean(data, axis = 0)
        spectrum   = np.mean(data, axis = 1)



    fig = plt.figure(figsize = (15,10))

    mpl.rcParams['axes.linewidth'] = 1.0

    plt.subplots_adjust(top = 0.99 , bottom = 0.1, right = 0.99, left = 0.1)

    widths =  [0.8,0.2]
    heights = [0.2,0.8]
    gs = plt.GridSpec(2,2,hspace = 0.0 , wspace = 0,  width_ratios = widths, height_ratios = heights)

    ax00 = plt.subplot(gs[0,0])
    ax10 = plt.subplot(gs[1,0])
    ax11 = plt.subplot(gs[1,1])

    ax00.set_xticks([])
    ax00.set_yticks([])
    ax11.set_xticks([])
    ax11.set_yticks([])

    size = 15

    ax10.margins(x = 0)
    ax11.margins(y = 0)
    ax10.tick_params(labelsize  = size)
    ax10.tick_params(labelsize  = size)
    ax10.set_ylabel("Frequency (MHz)", size = size)
    ax10.set_xlabel("Time (s)" , size = size)

    fig.align_labels()

    ax10.imshow(data , aspect = "auto", extent = (time[0], time[-1], freqs[-1], freqs[0]))
    ax00.plot(time, timeseries, color = "black")
    ax11.plot(spectrum,channels)


    if (save_flag is False):
        plt.show()
    else:
        output_name = output_name + file_format
        plt.savefig(os.path.join(output_dir, output_name))



def _get_parser():
    """
    Argument parser.
    """

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description = "Read a SIGPROC filterbank file and plot a portion of the data")
    parser.add_argument('-f',
                        '--fil_file',
                        action = "store" ,
                        help = "SIGPROC .fil file to be processed (REQUIRED).",
                        required = True)
    parser.add_argument('-o',
                        '--output_dir',
                        action = "store" ,
                        help = "Output directory (Default: your current path).",
                        default = "%s/"%(os.getcwd())
                        )
    parser.add_argument('-n',
                        '--output_name',
                        action = "store" ,
                        help = "Output File Name (Default: filename_cleaned.fil).",
                        default = None
                        )
    parser.add_argument('-s',
                        '--save_data',
                        help = "Save the candidate plot. (Default = True).",
                        action = 'store_true',
                        )
    parser.add_argument('-g',
                        '--grab_data',
                        help = "Grab a portion of the data (Default = False).",
                        action = 'store_false',
                        )
    parser.add_argument('-ff',
                        '--file_format',
                        action = "store" ,
                        help = "Format of the candidate image (Default: .png).",
                        default = None
                        )

    return parser.parse_args()


if __name__ == '__main__':

    args = _get_parser()

    filename    = args.fil_file
    output_dir  = args.output_dir
    output_name = args.output_name
    save_flag   = args.save_data
    grab_flag   = args.grab_data
    fileformat  = args.file_format

    plot_data(filename,
                       output_dir = output_dir,
                       output_name = output_name,
                       grab_flag = grab_flag,
                       save_flag = save_flag,
                       file_format = fileformat,
                       tstart = 0,
                       tstop  = 1
                       )
