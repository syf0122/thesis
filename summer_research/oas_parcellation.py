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
    for i in range(361):
        p[i] = []
    for x in range(data_mtx.shape[0]):
        for y in range(data_mtx.shape[1]):
            for z in range(data_mtx.shape[2]):
                ind = int(round(atlas[x][y][z]))
                if x > threshold: # right hemisphere
                    ind += 180
                p[ind].append(data_mtx[x][y][z]) # 149 timepoints
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

atlas_2 = nib.load("/data_qnap/yifeis/NAS/data/mmp_atlas/HCP-MMP1_on_MNI152_ICBM2009a_nlin_2mm_cropped.nii.gz")
atlas_2_data = atlas_2.get_fdata()
atlas_4 = nib.load("/data_qnap/yifeis/NAS/data/mmp_atlas/HCP-MMP1_on_MNI152_ICBM2009a_nlin_4mm_cropped.nii.gz")
atlas_4_data = atlas_4.get_fdata()
# get all subjects
dir = "/data_qnap/yifeis/NAS/data/AD/"
subjects = os.listdir(dir)
subjects.sort()

volume_files  = {} # volume file directory for each subject
volume_data   = {} # volume data for each subject
surface_files = {} # surface file directory for each subject
surface_data  = {} # surface data for each subject
volume_files_4 = {}
volume_data_4 = {}
print("There are " + str(len(subjects))+ " subjects under " + dir) # 443 subjects

# get directories of files and load data
for subject in subjects[50:60]:
    volume = [] # volume file directory
    v_data   = [] # volume data
    surface = [] # volume file directory
    s_data   = [] # volume data
    volume_4 = []
    v_data_4 = []
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
            # find volume data
            if "_space-MNI152NLin6Asym_desc-smoothAROMAnonaggr_bold.nii.gz" in f:
                d = nib.load(func_dir+f)    # load data
                if d.shape[-1] <= 15:
                    continue
                else:
                    v_data.append(d.get_fdata()[:, :, :, 15:])
                    if f[33:38][0] == "r":
                        volume.append(f[33:38]) # save the count of run
                    else:
                        volume.append("run")
            elif "91k_bold.dtseries.nii" in f:
                d = nib.load(func_dir+f)    # load data
                if d.shape[0] <= 15:
                    continue
                else:
                    s_data.append(d.get_fdata()[15:, :])
                    if f[33:38][0] == "r":
                        surface.append(f[33:38]) # save the count of run
                    else:
                        surface.append("run")

            elif "space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz" in f:
                d = nib.load(func_dir+f)    # load data
                if d.shape[-1] <= 15:
                    continue
                else:
                    v_data_4.append(d.get_fdata()[:, :, :, 15:])
                    if f[33:38][0] == "r":
                        volume_4.append(f[33:38]) # save the count of run
                    else:
                        volume_4.append("run")

    volume_files[subject] = volume
    volume_data[subject] = v_data
    surface_files[subject] = surface
    surface_data[subject] = s_data
    volume_data_4[subject] = v_data_4
    volume_files_4[subject] = volume_4


######## smoothed volume data (2mm)
# parcellate the volume data for subjects
vol_p_data = {}
for sub in subjects[50:60]:
    sub_data = volume_data[sub]
    sub_p_ls = []
    for i in range(len(sub_data)):
        p = parcellate_data(sub_data[i], atlas_2_data, 45)
        sub_p_ls.append(p)
        # np.save('/data_qnap/yifeis/NAS/parcellation/OAS/'+str(sub)+'_'+volume_files[sub][i]+'_2mm_vol_p.npy', p)
    vol_p_data[sub] = sub_p_ls

# check the shape of the data
for n in vol_p_data:
    print(n)
    print(len(vol_p_data[n]))
    print(vol_p_data[n][0].shape)
    print(vol_p_data[n][1].shape)

"""
    Correlation + Save plots
"""
correlation_measure = ConnectivityMeasure(kind='correlation')
sub_fc_mtx = {}
for sub in subjects[50:60]:
    sub_p = vol_p_data[sub]
    sub_fc_ls = []
    for i in range(len(sub_p)):
        mtx = correlation_measure.fit_transform([sub_p[i]])[0]
        nlp.plot_matrix(mtx, figure=(8, 6), reorder=False, cmap='jet')
        plt.title(sub + " " + volume_files[sub][i] + " 2mm Volume Raw Data Correlation Matrix")
        plt.savefig('/data_qnap/yifeis/NAS/parcellation/OAS/AD/' + sub + "_" + volume_files[sub][i] + "_2mm_vol_Correlation_Matrix.png", bbox_inches='tight')
        sub_fc_ls.append(mtx)
    sub_fc_mtx[sub] = (sub_fc_ls)

# check the shape of the data
for n in sub_fc_mtx:
    print(n)
    print(len(sub_fc_mtx[n]))
    print(sub_fc_mtx[n][0].shape)
    print(sub_fc_mtx[n][1].shape)

###### surface_data
# parcellate the surface data for subjects
surf_p_data = {}
for sub in subjects[50:60]:
    sub_data = surface_data[sub]
    sub_p_ls = []
    for i in range(len(sub_data)):
        p = hcp.parcellate(sub_data[i], hcp.mmp)
        sub_p_ls.append(p)
        # np.save('/data_qnap/yifeis/NAS/parcellation/OAS/'+str(sub)+'_'+surface_files[sub][i]+'_surf_p.npy', p)
    surf_p_data[sub] = sub_p_ls
# check the shape of the data
for n in surf_p_data:
    print(n)
    print(len(surf_p_data[n]))
    print(surf_p_data[n][0].shape)
    print(surf_p_data[n][1].shape)
"""
    Correlation + Save plots
"""
correlation_measure = ConnectivityMeasure(kind='correlation')
sub_fc_mtx = {}
for sub in subjects[50:60]:
    sub_p = surf_p_data[sub]
    sub_fc_ls = []
    for i in range(len(sub_p)):
        mtx = correlation_measure.fit_transform([sub_p[i]])[0]
        nlp.plot_matrix(mtx, figure=(8, 6), reorder=False, cmap='jet')
        plt.title(sub + " " + surface_files[sub][i] + " Surface Raw Data Correlation Matrix")
        plt.savefig('/data_qnap/yifeis/NAS/parcellation/OAS/AD/' + sub + "_" + surface_files[sub][i] + "_surf_Correlation_Matrix.png", bbox_inches='tight')
        sub_fc_ls.append(mtx)
    sub_fc_mtx[sub] = (sub_fc_ls)

######## preprocessed volume data (4mm)
# parcellate the volume data for subjects
vol_p_data_4 = {}
for sub in subjects[50:60]:
    sub_data = volume_data_4[sub]
    sub_p_ls = []
    for i in range(len(sub_data)):
        p = parcellate_data(sub_data[i], atlas_4_data, 25)
        sub_p_ls.append(p)
        # np.save('/data_qnap/yifeis/NAS/parcellation/OAS/'+str(sub)+'_'+volume_files_4[sub][i]+'_4mm_vol_p.npy', p)
    vol_p_data_4[sub] = sub_p_ls

# check the shape of the data
for n in vol_p_data_4:
    print(n)
    print(len(vol_p_data_4[n]))
    print(vol_p_data_4[n][0].shape)
    print(vol_p_data_4[n][1].shape)

"""
    Correlation + Save plots
"""
correlation_measure = ConnectivityMeasure(kind='correlation')
sub_fc_mtx_4 = {}
for sub in subjects[50:60]:
    sub_p = vol_p_data_4[sub]
    sub_fc_ls = []
    for i in range(len(sub_p)):
        mtx = correlation_measure.fit_transform([sub_p[i]])[0]
        nlp.plot_matrix(mtx, figure=(8, 6), reorder=False, cmap='jet')
        plt.title(sub + " " + volume_files_4[sub][i] + " 4mm Volume Raw Data Correlation Matrix")
        plt.savefig('/data_qnap/yifeis/NAS/parcellation/OAS/AD/' + sub + "_" + volume_files_4[sub][i] + "_4mm_vol_Correlation_Matrix.png", bbox_inches='tight')
        sub_fc_ls.append(mtx)
    sub_fc_mtx_4[sub] = (sub_fc_ls)

# check the shape of the data
for n in sub_fc_mtx_4:
    print(n)
    print(len(sub_fc_mtx_4[n]))
    print(sub_fc_mtx_4[n][0].shape)
    print(sub_fc_mtx_4[n][1].shape)
