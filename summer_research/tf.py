import sys
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import nibabel as nib
import hcp_utils as hcp
from nilearn import plotting as nlp
import multiprocessing
from multiprocessing import Process
# from nilearn.plotting.cm import _cmap_d as cm
# from nilearn import surface as nsf
# from nilearn.connectome import ConnectivityMeasure

def bandpass_filter(img, TR=2.2):
    data = img.get_fdata()[:, :, :, 15:]
    hp_freq = 0.01 # Hz
    lp_freq = 0.1 # Hz
    fs = 1 / TR # sampling rate, TR in s, fs in Hz
    timepoints = data.shape[-1]
    F = np.zeros(timepoints)
    # print(timepoints)
    lowidx = timepoints // 2 + 1
    if lp_freq > 0: # map cutoff frequencies to corresponding index
        lowidx = int(np.round(lp_freq / fs * timepoints))
    highidx = 0
    if hp_freq > 0: # map cutoff frequencies to corresponding index
        highidx = int(np.round(hp_freq / fs * timepoints))
    F[highidx:lowidx] = 1 # let signal with freq between lower and upper bound pass and remove others
    F = ((F + F[::-1]) > 0).astype(int)
    filtered_data = np.zeros(data.shape)
    # print(data.shape)
    if np.all(F == 1):
        filtered_data = data
    else:
        filtered_data = np.real(np.fft.ifftn(np.fft.fftn(data) * F))
    return filtered_data

# get all subjects
dir = "/data_qnap/yifeis/NAS/data/HC/"
subjects = os.listdir(dir)
subjects.sort()

smoothed_files  = {} # smoothed file directory for each subject
prepro_files    = {} # preprocessed file directory for each subject
smoothed_data   = {} # smoothed data for each subject
prepro_data     = {} # preprocessed data for each subject
print("There are " + str(len(subjects))+ " subjects under " + dir)

# get directories of files and load data
for subject in subjects[100:120]:
    smoothed = [] # smoothed file directory 2mm
    prepro   = [] # preprocessed file directory 4mm
    s_data   = [] # smoothed data
    p_data   = [] # preprocessed data

    # get the directory of to the functional data of this subject
    subject_dir = os.listdir(dir+subject)
    func_dir = None
    for d in subject_dir:
        if "ses-" in d:
            func_dir = dir+subject+"/"+d+"/func/"

    if func_dir == None:
        print("No such directory for subject " + subject)
    else:
        func_files = os.listdir(func_dir)
        func_files.sort()
        for f in func_files:
            # find smoothed data
            if "_space-MNI152NLin6Asym_desc-smoothAROMAnonaggr_bold.nii.gz" in f:
                d = nib.load(func_dir+f)    # load data
                if d.shape[-1] <= 50:
                    continue
                else:
                    s_data.append(d)
                    if f[33:38][0] == "r":
                        smoothed.append(f[33:38]) # save the count of run
                    else:
                        smoothed.append("run")

            # find preprocessed data
            elif "_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz" in f:
                d = nib.load(func_dir+f)  # load data
                if d.shape[-1] <= 50:
                    continue
                else:
                    p_data.append(d)
                    if f[33:38][0] == "r":
                        prepro.append(f[33:38]) # save the count of run
                    else:
                        prepro.append("run")

    smoothed_files[subject] = smoothed
    prepro_files[subject] = prepro
    smoothed_data[subject] = s_data
    prepro_data[subject] = p_data

sub_1   = subjects[100:102]
sub_2   = subjects[102:104]
sub_3   = subjects[104:106]
sub_4   = subjects[106:108]
sub_5   = subjects[108:110]
sub_6   = subjects[110:112]
sub_7   = subjects[112:114]
sub_8   = subjects[114:116]
sub_9   = subjects[116:118]
sub_10  = subjects[118:120]
"""
    temporal filtering
"""
def tf_multi_subs (subjects, given_data, given_files):
    sm_tf_data = {}
    for sub in subjects:
        sm_data = given_data[sub]
        sm_tf_ls = []
        for i in range(len(sm_data)):
            img = sm_data[i]
            session = given_files[sub][i]
            tf_data = bandpass_filter(img, 2.2)
            sm_tf_ls.append(tf_data)
            print(tf_data.shape)
            tf_img = nib.Nifti1Image(tf_data, img.affine)
            nib.save(tf_img, '/data_qnap/yifeis/NAS/data/HC_tf/'+str(sub)+'_'+ str(session)+ "_"+ 'temporal_filtered_4mm_bold.nii.gz')
        sm_tf_data[sub] = sm_tf_ls
        print(sub + " Finished!")


# multithreading
print("Start multiprocessing!") #smoothed_data
p1  = Process(target=tf_multi_subs, args=(sub_1, prepro_data, prepro_files))
p2  = Process(target=tf_multi_subs, args=(sub_2, prepro_data, prepro_files))
p3  = Process(target=tf_multi_subs, args=(sub_3, prepro_data, prepro_files))
p4  = Process(target=tf_multi_subs, args=(sub_4, prepro_data, prepro_files))
p5  = Process(target=tf_multi_subs, args=(sub_5, prepro_data, prepro_files))
p6  = Process(target=tf_multi_subs, args=(sub_6, prepro_data, prepro_files))
p7  = Process(target=tf_multi_subs, args=(sub_7, prepro_data, prepro_files))
p8  = Process(target=tf_multi_subs, args=(sub_8, prepro_data, prepro_files))
p9  = Process(target=tf_multi_subs, args=(sub_9, prepro_data, prepro_files))
p10 = Process(target=tf_multi_subs, args=(sub_10, prepro_data, prepro_files))

processes = [p1, p2, p3, p4, p5,
             p6, p7, p8, p9, p10]
for p in processes:
    p.start()
for p in processes:
    p.join()
