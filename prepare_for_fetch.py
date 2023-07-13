import sys
import os
import numpy as np
import pandas as pd
import argparse

def make_for_fetch(filfile,candfile,mask,outdir):

    columns = ["snr" , "tcand", "ncand", "dm", "filter", "beam"]
    #df = pd.read_csv(candfile, delimiter="\t", names = columns)

    df = pd.read_csv(candfile, delimiter=r"\s+", names = columns)#pd.read_csv(candfile) #pd.read_csv(candfile, delimiter=" ", names = columns)

    snr    = df["snr"].to_numpy()
    tcand  = df["tcand"].to_numpy()
    ncand  = df["ncand"].to_numpy()
    dm     = df["dm"].to_numpy()
    filter = df["filter"].to_numpy()

    head, tail = os.path.split(candfile)
    candroot = tail.replace('.good_cand', '')

    filename = ("%s/%s_forfetch.csv"%(outdir , candroot))

    filecsv = open(filename, "w")

    filecsv.write("file,snr,stime,width,dm,label,chan_mask_path,num_files\n")

    for lineidx in range(snr.shape[0]):
        filecsv.write("%s,%.5f,%.5f,%0d,%.5f,0,%s,1\n"%(filfile, snr[lineidx], tcand[lineidx], filter[lineidx], dm[lineidx], mask))

    filecsv.close()




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = "Create a .csv file ready for FETCH from an Heimdall candidate file processed with Ben's frb_detector.py")

    parser.add_argument('-f', '--fil_file',   action = "store" ,  help = "Path of the SIGPROC .fil file", required = True)
    parser.add_argument('-m', '--mask_file',   action = "store" , help = "Path of the mask file", default = " ")
    parser.add_argument('-c', '--cand_file',   action = "store" , help = "Path of the candidate file obtained from Heimdall's coincidencer", required = True)
    parser.add_argument('-o', '--output_dir', action = "store" , help = "Output for FETCH (Default: your current path)", default = "%s/"%(os.getcwd()))

    args = parser.parse_args()

    filfile  = args.fil_file
    candfile = args.cand_file
    mask     = args.mask_file
    outdir   = args.output_dir

    make_for_fetch(filfile,candfile,mask,outdir)
