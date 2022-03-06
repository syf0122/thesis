import os
import glob
import sys
import threading
import _thread as thread
import time
from pathlib import Path
import seaborn as sns
import numpy as np
import multiprocessing
from multiprocessing import Process
import pandas as pd
from matplotlib import pyplot as plt
import nibabel as nib
import hcp_utils as hcp
from nilearn import plotting as nlp
from nilearn import surface as nsf

def calculate_corr(df, ldf, subject, session, ntp, loc):
    """
        calculate the correlation matrix of the two input data
    """
    print(df.shape[1] == 91282)
    print(ldf.shape[1] == ntp)
    print(df.shape[0] == ldf.shape[0])
    cor_mtx = []
    for n in range(ntp):
        lat = []
        for m in range(91282):
            lat.append(ldf[str(n)].corr(df[m]))
        cor_mtx.append(lat)
    cor_df = pd.DataFrame(cor_mtx)
    cor_df.to_csv("/data_qnap/yifeis/ae_relu_trained_with_50/corr_" + loc + subject + "_" + session + "_corr.csv")
    print(session + " done!")
    return cor_df

"""
    load the latent data
    load the preprocessed data
    get data directories

    for each subject, there are 14 fmri sessions
    preprocessed data: (# of timepoints, 91292)
    latent data: (# of timepoints, tp)
    The order of data sessions:
"""
# get all the subjects
subjects = os.listdir("/data_qnap/yifeis/new/processed")
subjects.sort()
subjects = subjects[:50]

# set
sub_1  = subjects[0:2]
sub_2  = subjects[2:4]
sub_3  = subjects[4:6]
sub_4  = subjects[6:8]
sub_5  = subjects[8:10]
sub_6  = subjects[10:12]
sub_7  = subjects[12:14]
sub_8  = subjects[14:16]
sub_9  = subjects[16:18]
sub_10 = subjects[18:20]
sub_11 = subjects[20:22]
sub_12 = subjects[22:24]
sub_13 = subjects[24:26]
sub_14 = subjects[26:28]
sub_15 = subjects[28:30]
sub_16 = subjects[30:32]
sub_17 = subjects[32:34]
sub_18 = subjects[34:36]
sub_19 = subjects[36:38]
sub_20 = subjects[38:40]
sub_21 = subjects[40:42]
sub_22 = subjects[42:44]
sub_23 = subjects[44:46]
sub_24 = subjects[46:48]
sub_25 = subjects[48:50]

ses = sys.argv[1]
loc = sys.argv[2]
if '80' in loc:
    tp = 80
elif '160' in loc:
    tp = 160
else:
    tp = 40
print(tp)
print("Calculate and save the matrix plot of " + ses + " sessions data from " +loc+".")
print("/data_qnap/yifeis/ae_relu_trained_with_50/"+loc)
# load the preprocessed data
d_dir = "/data_qnap/yifeis/new/processed/"
vox_data_dir = {}
vox_data = {}
for sub in subjects:
    sub_dir = d_dir + sub + "/"
    tf_data_dir = []
    tf_data = []
    all_files = os.listdir(sub_dir)
    all_files.sort()
    for f in all_files:
        if "temporal_filtered_" in f and ses.upper() in f:
            tf_data_dir.append(sub_dir+f)
            data = nib.load(sub_dir+f)
            data = data.get_fdata(dtype=np.float32)
            data = hcp.normalize(data)
            df = pd.DataFrame(data)
            print(df.shape) # (# of timepoints, 91282)
            tf_data.append(df)
    vox_data_dir[sub] = tf_data_dir
    vox_data[sub] = tf_data
for n in vox_data:
    if len(vox_data[n]) != 1:
        print(n)
        print(len(vox_data[n]))
print("Voxel based data loaded")

# load latent data
latent_data_dir = {}
latent_data = {}
for sub in subjects:
    latent_data_dir[sub] = []
    latent_data[sub] = []
latent_dir = os.listdir("/data_qnap/yifeis/ae_relu_trained_with_50/"+loc)
latent_dir.sort()
for dir in latent_dir:
    for sub in subjects:
        if sub in dir and ses.lower() in dir:
            latent_data_dir[sub].append("/data_qnap/yifeis/ae_relu_trained_with_50/"+loc+dir)
            df = pd.read_csv("/data_qnap/yifeis/ae_relu_trained_with_50/"+loc+dir,index_col=0)
            print(df.shape) # (# of timepoints, tp)
            latent_data[sub].append(df)
for n in latent_data:
    if len(latent_data[n]) != 1:
        print(n)
        print(len(latent_data[n]))
print("Latent data loaded")

"""
    calculate the correlation between voxel data and latent data
    save the matrix plot
"""
def several_sub_correlation(sub_ls, vox_data, latent_data, ses, ntp, loc):
    for sub in sub_ls:
        print(sub + " starts!")
        vox_all = vox_data[sub]
        latent_all = latent_data[sub]
        print(len(vox_all) == 1)
        print(len(vox_all) == len(latent_all))
        for i in range(len(latent_all)):
            vox_df = vox_all[i]
            latent_df = latent_all[i]
            if len(vox_all) > 1:
                sub_ses = ses.lower()+str(i+1)
            elif len(vox_all) == 1:
                sub_ses = ses.lower()
            else:
                break
            corr_mtx = calculate_corr(vox_df, latent_df, sub, sub_ses, ntp, loc)
        print(sub + " finished!")

# multiprocessing
print("Start multiprocessing!")

p1 = Process(target=several_sub_correlation, args=(sub_1, vox_data, latent_data, ses, tp, loc))
p2 = Process(target=several_sub_correlation, args=(sub_2, vox_data, latent_data, ses, tp, loc))
p3 = Process(target=several_sub_correlation, args=(sub_3, vox_data, latent_data, ses, tp, loc))
p4 = Process(target=several_sub_correlation, args=(sub_4, vox_data, latent_data, ses, tp, loc))
p5 = Process(target=several_sub_correlation, args=(sub_5, vox_data, latent_data, ses, tp, loc))
p6 = Process(target=several_sub_correlation, args=(sub_6, vox_data, latent_data, ses, tp, loc))
p7 = Process(target=several_sub_correlation, args=(sub_7, vox_data, latent_data, ses, tp, loc))
p8 = Process(target=several_sub_correlation, args=(sub_8, vox_data, latent_data, ses, tp, loc))
p9 = Process(target=several_sub_correlation, args=(sub_9, vox_data, latent_data, ses, tp, loc))
p10 = Process(target=several_sub_correlation, args=(sub_10, vox_data, latent_data, ses, tp, loc))
p11 = Process(target=several_sub_correlation, args=(sub_11, vox_data, latent_data, ses, tp, loc))
p12 = Process(target=several_sub_correlation, args=(sub_12, vox_data, latent_data, ses, tp, loc))
p13 = Process(target=several_sub_correlation, args=(sub_13, vox_data, latent_data, ses, tp, loc))
p14 = Process(target=several_sub_correlation, args=(sub_14, vox_data, latent_data, ses, tp, loc))
p15 = Process(target=several_sub_correlation, args=(sub_15, vox_data, latent_data, ses, tp, loc))
p16 = Process(target=several_sub_correlation, args=(sub_16, vox_data, latent_data, ses, tp, loc))
p17 = Process(target=several_sub_correlation, args=(sub_17, vox_data, latent_data, ses, tp, loc))
p18 = Process(target=several_sub_correlation, args=(sub_18, vox_data, latent_data, ses, tp, loc))
p19 = Process(target=several_sub_correlation, args=(sub_19, vox_data, latent_data, ses, tp, loc))
p20 = Process(target=several_sub_correlation, args=(sub_20, vox_data, latent_data, ses, tp, loc))
p21 = Process(target=several_sub_correlation, args=(sub_21, vox_data, latent_data, ses, tp, loc))
p22 = Process(target=several_sub_correlation, args=(sub_22, vox_data, latent_data, ses, tp, loc))
p23 = Process(target=several_sub_correlation, args=(sub_23, vox_data, latent_data, ses, tp, loc))
p24 = Process(target=several_sub_correlation, args=(sub_24, vox_data, latent_data, ses, tp, loc))
p25 = Process(target=several_sub_correlation, args=(sub_25, vox_data, latent_data, ses, tp, loc))

processes = [p1, p2, p3, p4, p5,
             p6, p7, p8, p9, p10,
             p11, p12, p13, p14, p15,
             p16, p17, p18, p19, p20,
             p21, p22, p23, p24, p25]
for p in processes:
    p.start()
for p in processes:
    p.join()
