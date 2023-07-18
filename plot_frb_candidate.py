import numpy as np
import matplotlib.pyplot as plt
from sigpyproc.Filterbank import Filterbank
from sigpyproc.Readers import FilReader
import scipy
import textwrap
from scipy.signal import correlate
from scipy.linalg import toeplitz
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
from scipy.signal import detrend
import getpass
from datetime import datetime
from tqdm import tqdm
import os
import sys
import argparse
from scipy.optimize import curve_fit

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

def renormalize_data(array):

    renorm_data = np.copy(array)
    spec = renorm_data.mean(1)

    renorm_data -= spec[:, np.newaxis]

    baseline = np.mean(renorm_data, axis=0)

    renorm_data -= baseline

    return renorm_data

def gauss(x,a,x0,sigma):

    """
    Simple Gaussian Function. I use this to fit the data to get FRB width.
    """

    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def get_width(time,timeseries,k):

    """
    Function to get the width of a burst from a Gaussian fit. The width is taken as the Full Width at Tenth Maximum (FWTM) FWTM = 4.292 x sigma
    """
    A    = np.max(timeseries)
    tmax = time[np.argmax(timeseries)]
    par_time_opt,par_time_cov = curve_fit(gauss,time,timeseries, p0=[A,tmax,0.01])

    sigma_t = np.abs(par_time_opt[2])

    W = k * sigma_t
    Werr = k * np.sqrt(np.abs(par_time_cov[2,2]))

    return W, Werr

def get_snr(timeseries, wsamp):

    """
    Compute the integrated S/N of single pulse timeseries. It requires the width (in samples) of the burst. I use the equation from McLaughlin and Cordes 2003.
    """

    if wsamp == 0:
        wsamp = 1
    wsamp = int(wsamp/ 2)
    amax = np.argmax(timeseries)

    mask = np.ones(timeseries.shape[0], dtype = bool)
    mask[amax - wsamp : amax + wsamp ] = 0

    mu  = np.mean(timeseries[mask]) # mean off-burst
    std = np.std(timeseries[mask])  # rms off-burst
    if std == 0:
      std = 1e-4

    tmax = np.max(timeseries)
    if np.max(timeseries) == 0:
      tmax = 1e-4

    Weq = np.sum(timeseries[amax - wsamp : amax + wsamp ]) / tmax

    SNR = np.sum(timeseries - mu) / (std * np.sqrt(Weq))

    return SNR

def get_bandpass_onoff(wfall, wsamp):

    timeseries = np.nansum(wfall, axis = 0)

    argmax = np.argmax(timeseries)

    array1 = wfall[:,0 : argmax - wsamp // 2]
    array2 = wfall[:,argmax + wsamp // 2 : - 1]
    offpulsewfall = np.concatenate((array1, array2), axis=1)
    onpulsewfall  = wfall[:,argmax - wsamp // 2 : argmax + wsamp // 2]


    onpulsebpass  = np.nansum(onpulsewfall, axis = 1)
    offpulsebpass = np.nansum(offpulsewfall, axis = 1)

    return onpulsebpass, offpulsebpass

def DMT(dedispdata, freqs, dt, DM = 0, dmsteps = 256, ref_freq = "bottom"):

    #dedispdata = np.nan_to_num(dedispdata, nan=0)

    dmrange = 0.4 * DM

    DMs = np.linspace(-dmrange, dmrange, dmsteps)

    dmt = np.zeros((dmsteps, dedispdata.shape[1]))

    for k,dm in enumerate(DMs):

        data = dedisperse(dedispdata, dm, freqs, dt, ref_freq= ref_freq)
        dmt[k,:] = np.nansum(data, axis = 0)#data.mean(0)

    return DMs,dmt

def bin_freq_channels(data, fbin_factor = 1):
    num_chan = data.shape[0]
    if num_chan % fbin_factor != 0:
        raise ValueError("frequency binning factor `fbin_factor` should be even")
    data = np.nanmean(data.reshape((num_chan // fbin_factor, fbin_factor) + data.shape[1:]), axis=1)
    return data

def downsample_mask(badchans, newshape):

    """
    Downsample an RFI mask. It considers, as way of downsampling, group of channels and from them it takes as a new downsampled value the most common value between 0 or 1.
    """

    oldshape = badchans.shape[0]

    ratio = int(oldshape / newshape)

    downbadchans = np.zeros((newshape,), dtype = int)

    badchansn = np.zeros((oldshape,) , dtype = int)
    badchansn[badchans] = 1

    for k in range(newshape):

        values, counts = np.unique(badchansn[k * ratio : (k+1) * ratio], return_counts = True)
        ind = np.argmax(counts)
        downbadchans[k] = values[ind]

    downbadchans = np.asarray(downbadchans, dtype = bool)
    return downbadchans

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

def dm_delay_curve(tstart, freqs, t0, DM):

    times = np.ones(freqs.shape[0]) * tstart - t0

    for k,f in enumerate(freqs):
        dt = dispersion_delay(f, freqs[0], dms = DM)
        times[k] = times[k] + dt

    return times

def plot_candidate(filename,
    tcand = 0,
    dmcand = 0,
    output_dir = os.getcwd(),
    output_name = "candidate",
    format_file = ".png",
    save_flag = False,
    sk_flag = False,
    sk_sig = 3,
    twin = 100,
    fshape = None,
    tshape = None,
    grab_channels = False,
    channel_start = None,
    channel_stop = None,
    klt_clean = False,
    var_frac = 0.3,
    renorm_flag = True,
    width = 1, #ms
    snr = 10
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

    twin    = twin * 1e-3 # width in s
    nwin    = np.rint(twin / dt / 2).astype("int")

    if dmcand == 0:
        delay = 1
    else:
        delay = dispersion_delay(fbot, ftop, dms = dmcand)

    if delay <= twin:
        delay = twin

    ncand  = np.rint(tcand / dt).astype("int")
    ndelay = np.rint(delay / dt).astype("int")

    data = filterbank.readBlock(ncand - ndelay, 2 * ndelay)

    if (sk_flag is True):
        badchans = sk_filter(data[:, ndelay : -1].T, df, dt, sigma = sk_sig)

    if klt_clean:
        neig, ev, evecs, rfitemplate = klt(data, var_frac)
        data -=  rfitemplate

    if renorm_flag:
        data = renormalize_data(data)

    dedispdata = dedisperse(data, dmcand, freqs, dt)

    if grab_channels:
        cstart = int(channel_start)
        cstop  = int(channel_stop)
        if cstart < 0:
            cstart = 0
        if cstop > nchan:
            cstop = nchan
        data = data[cstart:cstop,:]
        dedispdata = dedispdata[cstart:cstop,:]
        freqs      = freqs[cstart:cstop]
        channels = channels[cstart:cstop]
        if (sk_flag is True):
            badchans = badchans[cstart:cstop]


    #Center the burst around a window (in ms)

    dedispdata = dedispdata[:, ndelay - nwin : ndelay + nwin]
    data = data[:, ndelay : -1]

    #data = detrend(data)
    #dedispdata = detrend(data)

    if (sk_flag is True):
        data[badchans, :] = np.nan
        dedispdata[badchans,:] = np.nan

    dms, dmt = DMT(dedispdata, freqs, dt, DM = dmcand, ref_freq = "top")

    time = np.linspace(-twin / 2, twin / 2, dedispdata.shape[1])
    timeseries = np.nansum(dedispdata, axis=0)

    #width, width_err = get_width(time,timeseries,2.355)


    if fshape is not None:
        data = resize(data, (fshape, data.shape[1]), anti_aliasing = True)
        dedispdata = resize(dedispdata, (fshape, data.shape[1]), anti_aliasing = True)
        freqs = np.linspace(freqs[0], freqs[-1], fshape)
        df = freqs[0] - freqs[1]
        if (sk_flag is True):
            badchans = downsample_mask(badchans, fshape)
    if tshape is not None:
        #print("data.shape (before):", data.shape)
        #print("dedispdata.shape (before):", dedispdata.shape)
        #print("dmt.shape: (before)", dmt.shape)
        #print("len(time): (before)", len(time))
        #print("dt: (before)", dt)
        data = resize(data, (data.shape[0], tshape), anti_aliasing = True)
        dedispdata = resize(dedispdata, (data.shape[0], tshape), anti_aliasing = True)
        dmt = resize(dmt, (dmt.shape[0], tshape), anti_aliasing = True)
        time = np.linspace(time[0], time[-1], tshape)
        dt = abs(time[0] - time[1])
        #print("data.shape (after):", data.shape)
        #print("dedispdata.shape (after):", dedispdata.shape)
        #print("dmt.shape: (after)", dmt.shape)
        #print("len(time): (after)", len(time))
        #print("dt: (after)", dt)


    timeseries = np.nansum(dedispdata, axis=0)
    wsamp = np.rint(width / dt).astype("int")
    #snr = get_snr(timeseries, wsamp)

    onbpass, offbpass = get_bandpass_onoff(dedispdata, wsamp)

    ondmcurve, offdmcurve = get_bandpass_onoff(dmt, wsamp)


    if (sk_flag is True):
        onbpass[badchans] = np.nan
        offbpass[badchans] = np.nan

    figure = plt.figure(figsize = (10,7))
    size = 12

    widths0  = [0.8,0.2]
    widths1  = [1]
    heights0 = [0.2,0.4,0.4]
    heights1 = [0.2,0.4,0.4]


    gs0  = plt.GridSpec(3,2,hspace = 0.0 , wspace = 0,  width_ratios = widths0, height_ratios = heights0, top = 0.99 , bottom = 0.1, right = 0.55, left = 0.10)
    gs1  = plt.GridSpec(3,1,hspace = 0.0 , wspace = 0,  width_ratios = widths1, height_ratios = heights1, top = 0.99 , bottom = 0.1, right = 0.99, left = 0.65)


    #ax0_01 = plt.subplot(gs0[0,1])
    ax0_00 = plt.subplot(gs0[0,0])
    ax0_10 = plt.subplot(gs0[1,0])
    ax0_11 = plt.subplot(gs0[1,1])
    ax0_20 = plt.subplot(gs0[2,0])
    #ax0_21 = plt.subplot(gs0[2,1])

    #ax1_00 = plt.subplot(gs1[0,0])
    #ax1_10 = plt.subplot(gs1[1,0])
    ax1_20 = plt.subplot(gs1[2,0])

    size = 15
    ax0_00.set_xticks([])
    ax0_00.set_yticks([])
    ax0_10.set_xticks([])
    ax0_11.set_xticks([])
    ax0_11.set_yticks([])
    #ax0_21.set_xticks([])
    #ax0_21.set_yticks([])
    ax0_00.margins(x=0)
    ax0_11.margins(y=0)
    #ax0_21.margins(y=0)

    #ax0_00.set_ylabel("S/N", size = size)
    ax0_10.set_ylabel("Frequency (MHz)", size = size)
    ax0_20.set_ylabel(r"DM (pc$\times$cm$^{-3}$)", size = size)
    ax0_20.set_xlabel("Time (ms)", size = size)

    ax1_20.set_ylabel("Frequency (MHz)", size = size)
    ax1_20.set_xlabel("Time (s)", size = size)

    #dedispdata[np.isnan(dedispdata)] = np.nanmedian(dedispdata)
    #data[np.isnan(data)] = np.nanmedian(data)

    ax0_00.plot(timeseries , color = "darkblue", linewidth = 2)
    vmin = np.nanpercentile(dedispdata, 1)
    vmax = np.nanpercentile(dedispdata, 99)
    ax0_10.imshow(dedispdata, aspect = "auto", extent = (-delay/2, delay/2, freqs[-1], freqs[0]),
                    cmap = "inferno")
    vmin = np.nanpercentile(data, 1)
    vmax = np.nanpercentile(data, 99)

    tp = dm_delay_curve(0, freqs, 0.0, dmcand)
    ax1_20.set_ylim(freqs[-1],freqs[0])
    #ax1_20.hlines(freqs[0], xmin = 0, xmax = delay, color = "lime", linewidth = 1)
    #ax1_20.hlines(freqs[-1],  xmin = 0, xmax = delay, color = "lime", linewidth = 1)
    #if grab_channels:
    #    ax1_20.hlines(freqs[cstart], xmin = 0, xmax = delay, color = "lime", linewidth = 1)
    #    ax1_20.hlines(freqs[cstop],  xmin = 0, xmax = delay, color = "lime", linewidth = 1)
    ax1_20.plot(tp, (1-0.02)*freqs, color = "white", linewidth = 1, linestyle = "dashed")
    ax1_20.plot(tp, (1+0.02)*freqs, color = "white", linewidth = 1, linestyle = "dashed")
    ax1_20.imshow(data, aspect = "auto", extent = (0, delay, freqs[-1], freqs[0]), cmap = "inferno")

    ax0_20.imshow(dmt, aspect = "auto", extent = (-twin * 1e3 / 2, twin * 1e3 / 2, dmcand + dms[0], dmcand + dms[-1]))

    #ax0_21.plot(offdmcurve, dmcand + dms , linewidth = 2, color = "darkred", alpha = 0.5)
    #ax0_21.plot(ondmcurve,  dmcand + dms , linewidth = 2, color = "darkgreen", alpha = 0.9)


    ax0_11.plot(offbpass, freqs, linewidth = 2, color = "darkred", alpha = 0.5)
    ax0_11.plot(onbpass,  freqs,linewidth = 2, color = "darkgreen", alpha = 0.9)

    max_line_width = 50
    wrapped_text = textwrap.fill(f"File directory: {filedir}", width=max_line_width)
    figure.text(0.650, 0.875, f"File directory: {wrapped_text}", fontsize=10)


    """
    figure.text(0.650,0.950, f"File Information" ,fontsize = 10)

    figure.text(0.650,0.900, f"File name: {name}" ,fontsize = 10)
    figure.text(0.650,0.875, f"File directory: {filedir}" ,fontsize = 310

    figure.text(0.650,0.825, f"Candidate Information" ,fontsize = 10)
    figure.text(0.650,0.800, f"Candidate arrival time (s) = {tcand}" ,fontsize = 10)
    figure.text(0.650,0.775, r"Candidate DM (pc$\times$cm$^{-3}$) = " + f"{dmcand}" ,fontsize = 10)
    figure.text(0.650,0.750, f"Candidate peak S/N = {snr:.2f}" ,fontsize = 10)
    figure.text(0.650,0.725, f"Candidate FWHM width (ms) = {width:.2f}" ,fontsize = 10)
    """

    username = getpass.getuser()
    datetimenow = datetime.utcnow()
    figure.text(0.650,0.01,"Plot made by %s on %s UTC"%(username,str(datetimenow)[0:19]), fontsize = 8)

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
        required = True,
    )
    parser.add_argument(
        "-w",
        "--width_cand",
        type = float,
        help = "Width of the candidate in ms.",
        default = 1,
    )
    parser.add_argument(
        "-snr",
        "--snr_cand",
        type = float,
        help = "S/N of the candidate.",
        default = 10,
    )
    parser.add_argument(
        "-tw",
        "--time_window",
        type = float,
        help = "Time window to grab and plot around the dedispersed burst (Default: 100 ms)",
        default = 100,
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
        "-nr",
        "--no_renorm",
        help="Do not renormalize the data",
        action="store_false",
    )
    parser.add_argument(
        "-ff",
        "--file_format",
        action="store",
        help="Format of the candidate image (Default: .png).",
        default="png",
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
    parser.add_argument('-fs',
                        '--f_shape',
                        type = int,
                        default = None,
                        action = "store" ,
                        help = "Shape of the data in frequency"
                        )
    parser.add_argument('-ts',
                        '--t_shape',
                        type = int,
                        default = None,
                        action = "store" ,
                        help = "Shape of the data in time"
                        )
    parser.add_argument(
                        "-c",
                        "--grab_channels",
                        help="Grab a portion of the data in frequency channels. Usage -c cstart cstop (Default = False).",
                        nargs=2,
                        type=int,
                        default=None,
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
    twin        = args.time_window
    sk_flag     = args.spectral_kurtosis
    sk_sig      = args.spectral_kurtosis_sigma
    fshape      = args.f_shape
    tshape      = args.t_shape
    grab_channels = args.grab_channels is not None
    channel_start, channel_stop = args.grab_channels or (None, None)
    klt_clean   = args.karhunen_loeve_cleaning
    var_frac    = args.variance_fraction
    renorm_flag = args.no_renorm
    width       = args.width_cand
    snr         = args.snr_cand


    plot_candidate(filename,
        tcand = tcand,
        dmcand = dmcand,
        output_dir = os.getcwd(),
        output_name = "candidate",
        format_file = "png",
        save_flag = save_flag,
        twin = twin,
        sk_flag = sk_flag,
        sk_sig = sk_sig,
        fshape = fshape,
        tshape = tshape,
        grab_channels= grab_channels,
        channel_start= channel_start,
        channel_stop= channel_stop,
        klt_clean = klt_clean,
        var_frac = var_frac,
        renorm_flag = renorm_flag,
        snr = snr,
        width = width
        )
