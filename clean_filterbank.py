import numpy as np
import matplotlib.pyplot as plt
from sigpyproc.Filterbank import Filterbank
from sigpyproc.Readers import FilReader
from scipy.signal import correlate
from scipy.linalg import toeplitz
from scipy.signal import savgol_filter
import sys

def elements_for_10_percent_sum(array):
    total_sum = np.sum(array)
    target_sum = 0.1 * total_sum  # 10% of the total sum
    sorted_indices = np.argsort(array)  # Sort the array indices in ascending order
    cumulative_sum = np.cumsum(array[sorted_indices])  # Calculate cumulative sum
    num_elements = np.searchsorted(cumulative_sum, target_sum, side='right') + 1
    return num_elements

def eigenbasis(A):
    evalues, evectors = np.linalg.eigh(A)

    sortidx  = evalues.argsort()[::-1]
    evalues  = evalues[sortidx]
    evectors = evectors[:,sortidx]

    return evalues,evectors

def klt(signal,neig):
    acf = correlate(signal-np.mean(signal),signal-np.mean(signal), mode='full')/len(signal)
    nmax = np.argmax(np.abs(acf))
    #acf = acf/acf[nmax]

    T = toeplitz(acf[nmax:nmax+len(signal)])

    eigenspectrum,eigenvectors = eigenbasis(T)

    coeff = np.matmul((signal[:]-np.mean(signal)),np.conjugate((eigenvectors[:,:])))

    recsignal = np.matmul(coeff[0:int(neig)],np.transpose(eigenvectors[:,0:int(neig)])) + np.mean(signal)

    return eigenspectrum,eigenvectors,recsignal

def iqr_filter(spec, smoothspec):

    detr = spec - smoothspec

    ordered = np.sort(detr)
    q1 = ordered[detr.size // 4]
    q2 = ordered[detr.size // 2]
    q3 = ordered[detr.size // 4 * 3]
    lowlim = q2 - 2 * (q2 - q1)
    hilim = q2 + 2 * (q3 - q2)

    badchans = (detr < lowlim) | (detr > hilim)

    return badchans

def read_and_clean(filename, outputname):

    """
    filename : name of the filterbankfile
    outputname : name of the filterbank cleaned
    """

    filterbank = FilReader(filename)

    nsamp = filterbank.header.nsamples
    nchan = filterbank.header.nchans
    nbits = filterbank.header.nbits

    outfile = filterbank.header.prepOutfile(outputname, back_compatible = True, nbits = nbits)

    data = filterbank.readBlock(0, nsamp) # (nchans, nsamp)

    datawrite = data.T.astype("uint8")
    #print(data.shape)

    outfile.cwrite(datawrite.ravel())



if __name__ == '__main__':

    filename = sys.argv[1]
    outputname = sys.argv[2]

    read_and_clean(filename, outputname)
