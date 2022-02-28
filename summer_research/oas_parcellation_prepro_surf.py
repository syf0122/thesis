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
from nilearn.plotting.cm import _cmap_d as cm
from nilearn import surface as nsf
from nilearn.connectome import ConnectivityMeasure

def bandpass_filter(mtx, TR):
    data = mtx.T
    hp_freq = 0.01 # Hz
    lp_freq = 0.1 # Hz
    fs = 1 / TR # sampling rate, TR in s, fs in Hz
    timepoints = data.shape[-1]
    F = np.zeros(timepoints)
    print(timepoints)
    lowidx = timepoints // 2 + 1
    if lp_freq > 0: # map cutoff frequencies to corresponding index
        lowidx = int(np.round(lp_freq / fs * timepoints))
    highidx = 0
    if hp_freq > 0: # map cutoff frequencies to corresponding index
        highidx = int(np.round(hp_freq / fs * timepoints))
    F[highidx:lowidx] = 1 # let signal with freq between lower and upper bound pass and remove others
    F = ((F + F[::-1]) > 0).astype(int) ### need double-check
    filtered_data = np.zeros(data.shape)
    print(data.shape)
    if np.all(F == 1):
        filtered_data = data
    else:
        filtered_data = np.real(np.fft.ifftn(np.fft.fftn(data) * F))
    return filtered_data.T

# get all subjects
dir = "/data_qnap/yifeis/NAS/data/AD/"
subjects = os.listdir(dir)
subjects.sort()

surface_files = {} # surface file directory for each subject
surface_data  = {} # surface data for each subject

# get directories of files and load data
for subject in subjects[60:100]:
    surface = [] # volume file directory
    s_data   = [] # volume data
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
            # find surface data
            if "91k_bold.dtseries.nii" in f:
                d = nib.load(func_dir+f)    # load data
                if d.shape[0] <= 50:
                    continue
                else:
                    s_data.append(d.get_fdata()[15:, :])
                    print(d.get_fdata()[15:, :].shape)
                    if f[33:38][0] == "r":
                        surface.append(f[33:38]) # save the count of run
                    else:
                        surface.append("run")
    surface_files[subject] = surface
    surface_data[subject] = s_data

###### surface_data
# temporal Filtering
tf_data = {}
for subject in subjects[60:100]:
    raw_data = surface_data[subject]
    tf_sub_ls = []
    for mtx in raw_data:
        tf_mtx = bandpass_filter(mtx, 2.2)
        tf_sub_ls.append(tf_mtx)
    tf_data[subject] = tf_sub_ls

# normalization
norm_data = {}
for subject in subjects[60:100]:
    data_ls = tf_data[subject]
    norm_sub_ls = []
    for mtx in data_ls:
        print(mtx.shape)
        norm_mtx = hcp.normalize(mtx)
        norm_sub_ls.append(norm_mtx)
    norm_data[subject] = norm_sub_ls

# save tf data after parcellation
surf_p_data = {}
for sub in subjects[60:100]:
    sub_data = tf_data[sub]
    sub_p_ls = []
    print(len(sub_data))
    for i in range(len(sub_data)):
        p = hcp.parcellate(sub_data[i], hcp.mmp)
        sub_p_ls.append(p)
        # np.save('/data_qnap/yifeis/NAS/data/AD_p/'+str(sub)+'_'+surface_files[sub][i]+'_surf_tf_p.npy', p)
    surf_p_data[sub] = sub_p_ls

# # check the shape of the data
# for n in surf_p_data:
#     print(n)
#     print(len(surf_p_data[n]))
#     print(surf_p_data[n][0].shape)
#     print(surf_p_data[n][1].shape)

"""
    Correlation + Save plots
"""
correlation_measure = ConnectivityMeasure(kind='correlation')
sub_fc_mtx = {}
for sub in subjects[60:100]:
    sub_p = surf_p_data[sub]
    sub_fc_ls = []
    for i in range(len(sub_p)):
        mtx = correlation_measure.fit_transform([sub_p[i]])[0]
        nlp.plot_matrix(mtx, figure=(8, 6), reorder=False, cmap='jet')
        plt.title(sub + " " + surface_files[sub][i] + " Surface Data Correlation Matrix (Temporal Filtered)")
        plt.savefig('/data_qnap/yifeis/NAS/parcellation/OAS/AD/' + sub + "_" + surface_files[sub][i] + "_surf_tf_Correlation_Matrix.png", bbox_inches='tight')
        sub_fc_ls.append(mtx)
    sub_fc_mtx[sub] = (sub_fc_ls)

# save nm data after parcellation
surf_p_data = {}
for sub in subjects[60:100]:
    sub_data = norm_data[sub]
    sub_p_ls = []
    print(len(sub_data))
    for i in range(len(sub_data)):
        p = hcp.parcellate(sub_data[i], hcp.mmp)
        sub_p_ls.append(p)
        # np.save('/data_qnap/yifeis/NAS/data/AD_p/'+str(sub)+'_'+surface_files[sub][i]+'_surf_norm_p.npy', p)
    surf_p_data[sub] = sub_p_ls
# # check the shape of the data
# for n in surf_p_data:
#     print(n)
#     print(len(surf_p_data[n]))
#     print(surf_p_data[n][0].shape)
#     print(surf_p_data[n][1].shape)

"""
    Correlation + Save plots
"""
correlation_measure = ConnectivityMeasure(kind='correlation')
sub_fc_mtx = {}
for sub in subjects[60:100]:
    sub_p = surf_p_data[sub]
    sub_fc_ls = []
    for i in range(len(sub_p)):
        mtx = correlation_measure.fit_transform([sub_p[i]])[0]
        nlp.plot_matrix(mtx, figure=(8, 6), reorder=False, cmap='jet')
        plt.title(sub + " " + surface_files[sub][i] + " Surface Data Correlation Matrix (Temporal Filtered and Normalized)")
        plt.savefig('/data_qnap/yifeis/NAS/parcellation/OAS/AD/' + sub + "_" + surface_files[sub][i] + "_surf_norm_Correlation_Matrix.png", bbox_inches='tight')
        sub_fc_ls.append(mtx)
    sub_fc_mtx[sub] = (sub_fc_ls)
