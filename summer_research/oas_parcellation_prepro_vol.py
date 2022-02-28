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

# get the label index for one volume data
def parcellate_data(data_mtx, atlas, threshold):
    p = {}
    print(data_mtx.shape)
    for i in range(361):
        p[i] = []
    for x in range(data_mtx.shape[0]):
        for y in range(data_mtx.shape[1]):
            for z in range(data_mtx.shape[2]):
                ind = int(round(atlas[x][y][z]))
                if x > threshold: # right hemisphere
                    ind += 180
                p[ind].append(data_mtx[x][y][z]) # 164 timepoints
    # at this point P is a dictionary for 361 regions
    # ignore the first region
    # calculate the averaged temporal data for each region
    avg_region = []
    for i in range(360):
        i = i + 1 # [1, 360]
        all_t_data = np.array(p[i])
        avg = np.mean(all_t_data, 0)
        avg_region.append(avg)
    avg_region = np.array(avg_region)
    avg_region = avg_region.T
    return avg_region

"""
    Load the atlas for processing
"""
atlas_2 = nib.load("/data_qnap/yifeis/NAS/data/mmp_atlas/HCP-MMP1_on_MNI152_ICBM2009a_nlin_2mm_cropped.nii.gz")
atlas_2_data = atlas_2.get_fdata()
atlas_4 = nib.load("/data_qnap/yifeis/NAS/data/mmp_atlas/HCP-MMP1_on_MNI152_ICBM2009a_nlin_4mm_cropped.nii.gz")
atlas_4_data = atlas_4.get_fdata()
# print("shape of the 2mm atlas: " + str(atlas_2_data.shape))
# print("shape of the 4mm atlas: " + str(atlas_4_data.shape))

"""
    Load temporal filtered data
"""
# get all subjects
subjects = os.listdir("/data_qnap/yifeis/NAS/data/AD/")
subjects.sort()

# get all temporal filtered data
tf_dir = "/data_qnap/yifeis/NAS/data/AD_tf/"
files = os.listdir(tf_dir)
files.sort()
tf_files = []
tf_files_4 = []

# 2mm
for f in files:
    if "2mm_bold.nii.gz" in f:
        tf_files.append(f)
    elif "4mm_bold.nii.gz" in f:
        tf_files_4.append(f)

# load the data
tf_data = {}
tf_session = {}
tf_data_4 = {}
tf_session_4 = {}
for sub in subjects[60:100]:
    sub_ls = []
    sub_ses = []
    sub_ls_4 = []
    sub_ses_4 = []
    for f in tf_files:
        if sub in f:
            data = nib.load(tf_dir + f)
            if data.shape[-1] <= 50:
                continue
            else:
                data = data.get_fdata()
                sub_ls.append(data)
                sub_ses.append(f[13:18])
    tf_data[sub] = sub_ls
    tf_session[sub] = sub_ses

    for f in tf_files_4:
        if sub in f:
            data = nib.load(tf_dir + f)
            if data.shape[-1] <= 50:
                continue
            else:
                data = data.get_fdata()
                sub_ls_4.append(data)
                sub_ses_4.append(f[13:18])
    tf_data_4[sub] = sub_ls_4
    tf_session_4[sub] = sub_ses_4

for n in tf_data:
    print(n)
    print(len(tf_data[n]))
    print(len(tf_data_4[n]))
    print(tf_data[n][0].shape)
    print(tf_data_4[n][0].shape)
    print()
print("TF")
"""
    Normalization
"""
# normalization
norm_data = {}
norm_data_4 = {}
for n in tf_data:
    # 2mm
    data_ls = tf_data[n]
    norm_sub_ls = []
    for mtx in data_ls:
        mtx = mtx.T
        mtx = hcp.normalize(mtx)
        mtx = mtx.T
        norm_sub_ls.append(mtx)
    norm_data[n] = norm_sub_ls

    # 4mm
    data_ls_4 = tf_data_4[n]
    norm_sub_ls_4 = []
    for mtx in data_ls_4:
        mtx = mtx.T
        mtx = hcp.normalize(mtx)
        mtx = mtx.T
        norm_sub_ls_4.append(mtx)
    norm_data_4[n] = norm_sub_ls_4

for n in tf_data:
    print(len(norm_data[n]))
    print(len(norm_data_4[n]))
    print(norm_data[n][0].shape)
    print(norm_data_4[n][0].shape)
    print()
print("Normalized")
"""
    Atlas
"""
# get all labels
labels = hcp.mmp.labels
l_labels = {}
r_labels = {}
for l in labels:
    if l >= 1 and l <= 180:
        l_labels[l] = labels[l]
    elif l >= 180 and l <= 360:
        r_labels[l] = labels[l]

# # check if the order is the same as the atls
# with open("/data_qnap/yifeis/NAS/data/AD_tf/HCP-MMP1_on_MNI152_ICBM2009a_nlin.txt") as f:
#     contents = f.readlines()
# f.close()
# mmp_labels_l = {}
# for c in contents:
#     x = c.split()
#     mmp_labels_l[int(x[0])] = x[1][:-4]
# print(mmp_labels_l == l_labels)

"""
    2mm/4mm normalization parcellation
"""
# parcellate the normalized data for subjects 2mm
sub_p_data = {}
sub_p_data_4 = {}
for sub in subjects[60:100]:
    # 2mm
    sub_data = norm_data[sub]
    sub_p_ls = []
    for i in range(len(sub_data)):
        p = parcellate_data(sub_data[i], atlas_2_data, 45)
        sub_p_ls.append(p)
        # np.save('/data_qnap/yifeis/NAS/data/AD_p/'+str(sub)+'_'+tf_session[sub][i]+'_2mm_norm_p.npy', p)
    sub_p_data[sub] = sub_p_ls

    # 4mm
    sub_data = norm_data_4[sub]
    sub_p_ls = []
    for i in range(len(sub_data)):
        p = parcellate_data(sub_data[i], atlas_4_data, 25)
        sub_p_ls.append(p)
        # np.save('/data_qnap/yifeis/NAS/data/AD_p/'+str(sub)+'_'+tf_session_4[sub][i]+'_4mm_norm_p.npy', p)
    sub_p_data_4[sub] = sub_p_ls

# # check the shape of the data
# for n in sub_p_data:
#     print(n)
#     print(len(sub_p_data[n]))
#     print(sub_p_data[n][0].shape)
#     print(sub_p_data[n][1].shape)
#     print(len(sub_p_data_4[n]))
#     print(sub_p_data_4[n][0].shape)
#     print(sub_p_data_4[n][1].shape)
print("norm saved")

# Correlation + Save plots
correlation_measure = ConnectivityMeasure(kind='correlation')
sub_fc_mtx = {}
sub_fc_mtx_4 = {}
for sub in subjects[60:100]:
    # 2mm
    sub_p = sub_p_data[sub]
    sub_fc_ls = []
    for i in range(len(sub_p)):
        mtx = correlation_measure.fit_transform([sub_p[i]])[0]
        nlp.plot_matrix(mtx, figure=(8, 6), reorder=False, cmap='jet')
        plt.title(sub + " " + tf_session[sub][i] + " 2mm Correlation Matrix (Temporal Filtered and Normalized)")
        plt.savefig('/data_qnap/yifeis/NAS/parcellation/OAS/AD/' + sub + "_" + tf_session[sub][i] + "_2mm_norm_Correlation_Matrix.png", bbox_inches='tight')
        sub_fc_ls.append(mtx)
    sub_fc_mtx[sub] = (sub_fc_ls)

    # 4mm
    sub_p = sub_p_data_4[sub]
    sub_fc_ls = []
    for i in range(len(sub_p)):
        mtx = correlation_measure.fit_transform([sub_p[i]])[0]
        nlp.plot_matrix(mtx, figure=(8, 6), reorder=False, cmap='jet')
        plt.title(sub + " " + tf_session_4[sub][i] + " 4mm Correlation Matrix (Temporal Filtered and Normalized)")
        plt.savefig('/data_qnap/yifeis/NAS/parcellation/OAS/AD/' + sub + "_" + tf_session_4[sub][i] + "_4mm_norm_Correlation_Matrix.png", bbox_inches='tight')
        sub_fc_ls.append(mtx)
    sub_fc_mtx_4[sub] = (sub_fc_ls)
#
# # check the shape of the data
# for n in sub_fc_mtx:
#     print(n)
#     print(len(sub_fc_mtx[n]))
#     print(sub_fc_mtx[n][0].shape)
#     print(sub_fc_mtx[n][1].shape)
#     print(n)
#     print(len(sub_fc_mtx_4[n]))
#     print(sub_fc_mtx_4[n][0].shape)
#     print(sub_fc_mtx_4[n][1].shape)


"""
    2mm/4mm tf parcellation
"""
# parcellate the normalized data for subjects 2mm
sub_p_data = {}
sub_p_data_4 = {}
for sub in subjects[60:100]:
    # 2mm
    sub_data = tf_data[sub]
    sub_p_ls = []
    for i in range(len(sub_data)):
        p = parcellate_data(sub_data[i], atlas_2_data, 45)
        sub_p_ls.append(p)
        # np.save('/data_qnap/yifeis/NAS/data/AD_p/'+str(sub)+'_'+tf_session[sub][i]+'_2mm_tf_p.npy', p)
    sub_p_data[sub] = sub_p_ls

    # 4mm
    sub_data = tf_data_4[sub]
    sub_p_ls = []
    for i in range(len(sub_data)):
        p = parcellate_data(sub_data[i], atlas_4_data, 25)
        sub_p_ls.append(p)
        # np.save('/data_qnap/yifeis/NAS/data/AD_p/'+str(sub)+'_'+tf_session_4[sub][i]+'_4mm_tf_p.npy', p)
    sub_p_data_4[sub] = sub_p_ls

# # check the shape of the data
# for n in sub_p_data:
#     print(n)
#     print(len(sub_p_data[n]))
#     print(sub_p_data[n][0].shape)
#     print(sub_p_data[n][1].shape)
#     print(len(sub_p_data_4[n]))
#     print(sub_p_data_4[n][0].shape)
#     print(sub_p_data_4[n][1].shape)
print("tf saved")

# Correlation + Save plots
correlation_measure = ConnectivityMeasure(kind='correlation')
sub_fc_mtx = {}
sub_fc_mtx_4 = {}
for sub in subjects[60:100]:
    # 2mm
    sub_p = sub_p_data[sub]
    sub_fc_ls = []
    for i in range(len(sub_p)):
        mtx = correlation_measure.fit_transform([sub_p[i]])[0]
        nlp.plot_matrix(mtx, figure=(8, 6), reorder=False, cmap='jet')
        plt.title(sub + " " + tf_session[sub][i] + " 2mm Correlation Matrix (Temporal Filtered)")
        plt.savefig('/data_qnap/yifeis/NAS/parcellation/OAS/AD/' + sub + "_" + tf_session[sub][i] + "_2mm_tf_Correlation_Matrix.png", bbox_inches='tight')
        sub_fc_ls.append(mtx)
    sub_fc_mtx[sub] = (sub_fc_ls)

    # 4mm
    sub_p = sub_p_data_4[sub]
    sub_fc_ls = []
    for i in range(len(sub_p)):
        mtx = correlation_measure.fit_transform([sub_p[i]])[0]
        nlp.plot_matrix(mtx, figure=(8, 6), reorder=False, cmap='jet')
        plt.title(sub + " " + tf_session_4[sub][i] + " 4mm Correlation Matrix (Temporal Filtered)")
        plt.savefig('/data_qnap/yifeis/NAS/parcellation/OAS/AD/' + sub + "_" + tf_session_4[sub][i] + "_4mm_tf_Correlation_Matrix.png", bbox_inches='tight')
        sub_fc_ls.append(mtx)
    sub_fc_mtx_4[sub] = (sub_fc_ls)
#
# # check the shape of the data
# for n in sub_fc_mtx:
#     print(n)
#     print(len(sub_fc_mtx[n]))
#     print(sub_fc_mtx[n][0].shape)
#     print(sub_fc_mtx[n][1].shape)
#     print(n)
#     print(len(sub_fc_mtx_4[n]))
#     print(sub_fc_mtx_4[n][0].shape)
#     print(sub_fc_mtx_4[n][1].shape)
