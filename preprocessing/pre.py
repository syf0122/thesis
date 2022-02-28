import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import nibabel as nib
import hcp_utils as hcp
from nilearn import plotting as nlp
from nilearn import surface as nsf
from nilearn.connectome import ConnectivityMeasure
from nipype.interfaces.workbench import CiftiSmooth

# ------ Functions ------ #
def volume_from_cifti(data, axis):
    assert isinstance(axis, nib.cifti2.BrainModelAxis)
    data = data.T[axis.volume_mask]
    volmask = axis.volume_mask
    vox_indices = tuple(axis.voxel[axis.volume_mask].T)
    vol_data =np.zeros(axis.volume_shape + data.shape[1:], dtype = data.dtype)
    vol_data[vox_indices] = data
    return vol_data

def surf_data_from_cifti(data, axis, surf_name):
    assert isinstance(axis, nib.cifti2.BrainModelAxis)
    for name, data_indices, model in axis.iter_structures():
        if name == surf_name:
            data = data.T[data_indices] # get the corresponding slice from the whole matrix
            vtx_indices = model.vertex
            # print((vtx_indices.max() + 1,)) # number of rows
            # print(data.shape[1:]) # number of columns
            surf_data = np.zeros((vtx_indices.max() + 1,) + data.shape[1:], dtype = data.dtype)
            surf_data[vtx_indices] = data
            return surf_data # numpy arrary
    raise ValueError(f"No structure named {surf_name}")

def decompose_cifti(img):
    data = img.get_fdata(dtype=np.float32) # extract fMRI timeseries to numpy arrary
    brain_models = img.header.get_axis(1)
    return (volume_from_cifti(data, brain_models),
            surf_data_from_cifti(data, brain_models, "CIFTI_STRUCTURE_CORTEX_LEFT"),
            surf_data_from_cifti(data, brain_models, "CIFTI_STRUCTURE_CORTEX_RIGHT"))

def bandpass_filter(img, TR):
    data = img.get_fdata(dtype=np.float32).T
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

# ------ load data ------ #
subject = sys.argv[1]
data_dir = Path('/data_qnap/yifeis/new/'+str(subject)+'/MNINonLinear/')
# movie
raw_mov1_dir = '/data_qnap/yifeis/new/'+str(subject)+'/MNINonLinear/Results/tfMRI_MOVIE1_7T_AP/tfMRI_MOVIE1_7T_AP_Atlas_MSMAll_hp2000_clean.dtseries.nii'
raw_mov2_dir = '/data_qnap/yifeis/new/'+str(subject)+'/MNINonLinear/Results/tfMRI_MOVIE2_7T_PA/tfMRI_MOVIE2_7T_PA_Atlas_MSMAll_hp2000_clean.dtseries.nii'
raw_mov3_dir = '/data_qnap/yifeis/new/'+str(subject)+'/MNINonLinear/Results/tfMRI_MOVIE3_7T_PA/tfMRI_MOVIE3_7T_PA_Atlas_MSMAll_hp2000_clean.dtseries.nii'
raw_mov4_dir = '/data_qnap/yifeis/new/'+str(subject)+'/MNINonLinear/Results/tfMRI_MOVIE4_7T_AP/tfMRI_MOVIE4_7T_AP_Atlas_MSMAll_hp2000_clean.dtseries.nii'
# rest
raw_rst1_dir = '/data_qnap/yifeis/new/'+str(subject)+'/MNINonLinear/Results/rfMRI_REST1_7T_PA/rfMRI_REST1_7T_PA_Atlas_MSMAll_hp2000_clean.dtseries.nii'
raw_rst2_dir = '/data_qnap/yifeis/new/'+str(subject)+'/MNINonLinear/Results/rfMRI_REST2_7T_AP/rfMRI_REST2_7T_AP_Atlas_MSMAll_hp2000_clean.dtseries.nii'
raw_rst3_dir = '/data_qnap/yifeis/new/'+str(subject)+'/MNINonLinear/Results/rfMRI_REST3_7T_PA/rfMRI_REST3_7T_PA_Atlas_MSMAll_hp2000_clean.dtseries.nii'
raw_rst4_dir = '/data_qnap/yifeis/new/'+str(subject)+'/MNINonLinear/Results/rfMRI_REST4_7T_AP/rfMRI_REST4_7T_AP_Atlas_MSMAll_hp2000_clean.dtseries.nii'
# task
raw_tsk1_dir = '/data_qnap/yifeis/new/'+str(subject)+'/MNINonLinear/Results/tfMRI_RETBAR1_7T_AP/tfMRI_RETBAR1_7T_AP_Atlas_MSMAll_hp2000_clean.dtseries.nii'
raw_tsk2_dir = '/data_qnap/yifeis/new/'+str(subject)+'/MNINonLinear/Results/tfMRI_RETBAR2_7T_PA/tfMRI_RETBAR2_7T_PA_Atlas_MSMAll_hp2000_clean.dtseries.nii'
raw_tsk3_dir = '/data_qnap/yifeis/new/'+str(subject)+'/MNINonLinear/Results/tfMRI_RETCCW_7T_AP/tfMRI_RETCCW_7T_AP_Atlas_MSMAll_hp2000_clean.dtseries.nii'
raw_tsk4_dir = '/data_qnap/yifeis/new/'+str(subject)+'/MNINonLinear/Results/tfMRI_RETCON_7T_PA/tfMRI_RETCON_7T_PA_Atlas_MSMAll_hp2000_clean.dtseries.nii'
raw_tsk5_dir = '/data_qnap/yifeis/new/'+str(subject)+'/MNINonLinear/Results/tfMRI_RETCW_7T_PA/tfMRI_RETCW_7T_PA_Atlas_MSMAll_hp2000_clean.dtseries.nii'
raw_tsk6_dir = '/data_qnap/yifeis/new/'+str(subject)+'/MNINonLinear/Results/tfMRI_RETEXP_7T_AP/tfMRI_RETEXP_7T_AP_Atlas_MSMAll_hp2000_clean.dtseries.nii'
# surface
surf_l_dir = '/data_qnap/yifeis/new/'+str(subject)+'/MNINonLinear/fsaverage_LR32k/'+str(subject)+'.L.midthickness_MSMAll.32k_fs_LR.surf.gii'
surf_r_dir = '/data_qnap/yifeis/new/'+str(subject)+'/MNINonLinear/fsaverage_LR32k/'+str(subject)+'.R.midthickness_MSMAll.32k_fs_LR.surf.gii'

movie1_cifti = nib.load(raw_mov1_dir)
movie2_cifti = nib.load(raw_mov2_dir)
movie3_cifti = nib.load(raw_mov3_dir)
movie4_cifti = nib.load(raw_mov4_dir)

rest1_cifti = nib.load(raw_rst1_dir)
rest2_cifti = nib.load(raw_rst2_dir)
rest3_cifti = nib.load(raw_rst3_dir)
rest4_cifti = nib.load(raw_rst4_dir)

tsk1_cifti = nib.load(raw_tsk1_dir)
tsk2_cifti = nib.load(raw_tsk2_dir)
tsk3_cifti = nib.load(raw_tsk3_dir)
tsk4_cifti = nib.load(raw_tsk4_dir)
tsk5_cifti = nib.load(raw_tsk5_dir)
tsk6_cifti = nib.load(raw_tsk6_dir)

# ------ Decompostition ------ #
m1vol, m1left, m1right = decompose_cifti(movie1_cifti) # surface data are stored as numpy arrary
m2vol, m2left, m2right = decompose_cifti(movie2_cifti)
m3vol, m3left, m3right = decompose_cifti(movie3_cifti)
m4vol, m4left, m4right = decompose_cifti(movie4_cifti)

r1vol, r1left, r1right = decompose_cifti(rest1_cifti) # row = spatial dimention, column = timepoints
r2vol, r2left, r2right = decompose_cifti(rest2_cifti)
r3vol, r3left, r3right = decompose_cifti(rest3_cifti)
r4vol, r4left, r4right = decompose_cifti(rest4_cifti)

t1vol, t1left, t1right = decompose_cifti(tsk1_cifti)
t2vol, t2left, t2right = decompose_cifti(tsk2_cifti)
t3vol, t3left, t3right = decompose_cifti(tsk3_cifti)
t4vol, t4left, t4right = decompose_cifti(tsk4_cifti)
t5vol, t5left, t5right = decompose_cifti(tsk5_cifti)
t6vol, t6left, t6right = decompose_cifti(tsk6_cifti)
# ------ Smoothing ------ #
# Spatial
# wb_command
ss_mov1_dir = '/data_qnap/yifeis/new/processed/'+str(subject)+'/spatial_smoothed_tfMRI_MOVIE1_7T_AP_Atlas_MSMAll_hp2000_clean.dtseries.nii'
ss_mov2_dir = '/data_qnap/yifeis/new/processed/'+str(subject)+'/spatial_smoothed_tfMRI_MOVIE2_7T_PA_Atlas_MSMAll_hp2000_clean.dtseries.nii'
ss_mov3_dir = '/data_qnap/yifeis/new/processed/'+str(subject)+'/spatial_smoothed_tfMRI_MOVIE3_7T_PA_Atlas_MSMAll_hp2000_clean.dtseries.nii'
ss_mov4_dir = '/data_qnap/yifeis/new/processed/'+str(subject)+'/spatial_smoothed_tfMRI_MOVIE4_7T_AP_Atlas_MSMAll_hp2000_clean.dtseries.nii'

ss_rst1_dir = '/data_qnap/yifeis/new/processed/'+str(subject)+'/spatial_smoothed_rfMRI_REST1_7T_PA_Atlas_MSMAll_hp2000_clean.dtseries.nii'
ss_rst2_dir = '/data_qnap/yifeis/new/processed/'+str(subject)+'/spatial_smoothed_rfMRI_REST2_7T_AP_Atlas_MSMAll_hp2000_clean.dtseries.nii'
ss_rst3_dir = '/data_qnap/yifeis/new/processed/'+str(subject)+'/spatial_smoothed_rfMRI_REST3_7T_PA_Atlas_MSMAll_hp2000_clean.dtseries.nii'
ss_rst4_dir = '/data_qnap/yifeis/new/processed/'+str(subject)+'/spatial_smoothed_rfMRI_REST4_7T_AP_Atlas_MSMAll_hp2000_clean.dtseries.nii'

ss_tsk1_dir = '/data_qnap/yifeis/new/processed/'+str(subject)+'/spatial_smoothed_tfMRI_RETBAR1_7T_AP_Atlas_MSMAll_hp2000_clean.dtseries.nii'
ss_tsk2_dir = '/data_qnap/yifeis/new/processed/'+str(subject)+'/spatial_smoothed_tfMRI_RETBAR2_7T_PA_Atlas_MSMAll_hp2000_clean.dtseries.nii'
ss_tsk3_dir = '/data_qnap/yifeis/new/processed/'+str(subject)+'/spatial_smoothed_tfMRI_RETCCW_7T_AP_Atlas_MSMAll_hp2000_clean.dtseries.nii'
ss_tsk4_dir = '/data_qnap/yifeis/new/processed/'+str(subject)+'/spatial_smoothed_tfMRI_RETCON_7T_PA_Atlas_MSMAll_hp2000_clean.dtseries.nii'
ss_tsk5_dir = '/data_qnap/yifeis/new/processed/'+str(subject)+'/spatial_smoothed_tfMRI_RETCW_7T_PA_Atlas_MSMAll_hp2000_clean.dtseries.nii'
ss_tsk6_dir = '/data_qnap/yifeis/new/processed/'+str(subject)+'/spatial_smoothed_tfMRI_RETEXP_7T_AP_Atlas_MSMAll_hp2000_clean.dtseries.nii'

def spat_smoothing(subject):
    ss_mov1_dir = '/data_qnap/yifeis/new/processed/'+str(subject)+'/spatial_smoothed_tfMRI_MOVIE1_7T_AP_Atlas_MSMAll_hp2000_clean.dtseries.nii'
    ss_mov2_dir = '/data_qnap/yifeis/new/processed/'+str(subject)+'/spatial_smoothed_tfMRI_MOVIE2_7T_PA_Atlas_MSMAll_hp2000_clean.dtseries.nii'
    ss_mov3_dir = '/data_qnap/yifeis/new/processed/'+str(subject)+'/spatial_smoothed_tfMRI_MOVIE3_7T_PA_Atlas_MSMAll_hp2000_clean.dtseries.nii'
    ss_mov4_dir = '/data_qnap/yifeis/new/processed/'+str(subject)+'/spatial_smoothed_tfMRI_MOVIE4_7T_AP_Atlas_MSMAll_hp2000_clean.dtseries.nii'

    ss_rst1_dir = '/data_qnap/yifeis/new/processed/'+str(subject)+'/spatial_smoothed_rfMRI_REST1_7T_PA_Atlas_MSMAll_hp2000_clean.dtseries.nii'
    ss_rst2_dir = '/data_qnap/yifeis/new/processed/'+str(subject)+'/spatial_smoothed_rfMRI_REST2_7T_AP_Atlas_MSMAll_hp2000_clean.dtseries.nii'
    ss_rst3_dir = '/data_qnap/yifeis/new/processed/'+str(subject)+'/spatial_smoothed_rfMRI_REST3_7T_PA_Atlas_MSMAll_hp2000_clean.dtseries.nii'
    ss_rst4_dir = '/data_qnap/yifeis/new/processed/'+str(subject)+'/spatial_smoothed_rfMRI_REST4_7T_AP_Atlas_MSMAll_hp2000_clean.dtseries.nii'

    ss_tsk1_dir = '/data_qnap/yifeis/new/processed/'+str(subject)+'/spatial_smoothed_tfMRI_RETBAR1_7T_AP_Atlas_MSMAll_hp2000_clean.dtseries.nii'
    ss_tsk2_dir = '/data_qnap/yifeis/new/processed/'+str(subject)+'/spatial_smoothed_tfMRI_RETBAR2_7T_PA_Atlas_MSMAll_hp2000_clean.dtseries.nii'
    ss_tsk3_dir = '/data_qnap/yifeis/new/processed/'+str(subject)+'/spatial_smoothed_tfMRI_RETCCW_7T_AP_Atlas_MSMAll_hp2000_clean.dtseries.nii'
    ss_tsk4_dir = '/data_qnap/yifeis/new/processed/'+str(subject)+'/spatial_smoothed_tfMRI_RETCON_7T_PA_Atlas_MSMAll_hp2000_clean.dtseries.nii'
    ss_tsk5_dir = '/data_qnap/yifeis/new/processed/'+str(subject)+'/spatial_smoothed_tfMRI_RETCW_7T_PA_Atlas_MSMAll_hp2000_clean.dtseries.nii'
    ss_tsk6_dir = '/data_qnap/yifeis/new/processed/'+str(subject)+'/spatial_smoothed_tfMRI_RETEXP_7T_AP_Atlas_MSMAll_hp2000_clean.dtseries.nii'
    os.mkdir('/data_qnap/yifeis/new/processed/'+str(subject))
    print("------Is Doing Spatial Smoothing.------")
    vol_size = 2
    fwhm = 6
    sigma = fwhm/(np.sqrt(8 * np.log(2)) * vol_size) # 1.27
    # rest
    smooth = CiftiSmooth(in_file = raw_rst1_dir,
                        sigma_surf = sigma,
                        sigma_vol = sigma,
                        direction = 'COLUMN',
                        right_surf = surf_r_dir,
                        left_surf = surf_l_dir,
                        out_file = ss_rst1_dir)
    os.system(smooth.cmdline)
    smooth = CiftiSmooth(in_file = raw_rst2_dir,
                        sigma_surf = sigma,
                        sigma_vol = sigma,
                        direction = 'COLUMN',
                        right_surf = surf_r_dir,
                        left_surf = surf_l_dir,
                        out_file = ss_rst2_dir)
    os.system(smooth.cmdline)
    smooth = CiftiSmooth(in_file = raw_rst3_dir,
                        sigma_surf = sigma,
                        sigma_vol = sigma,
                        direction = 'COLUMN',
                        right_surf = surf_r_dir,
                        left_surf = surf_l_dir,
                        out_file = ss_rst3_dir)
    os.system(smooth.cmdline)
    smooth = CiftiSmooth(in_file = raw_rst4_dir,
                        sigma_surf = sigma,
                        sigma_vol = sigma,
                        direction = 'COLUMN',
                        right_surf = surf_r_dir,
                        left_surf = surf_l_dir,
                        out_file = ss_rst4_dir)
    os.system(smooth.cmdline)

    # movie
    smooth = CiftiSmooth(in_file = raw_mov1_dir,
                        sigma_surf = sigma,
                        sigma_vol = sigma,
                        direction = 'COLUMN',
                        right_surf = surf_r_dir,
                        left_surf = surf_l_dir,
                        out_file = ss_mov1_dir)
    os.system(smooth.cmdline)
    smooth = CiftiSmooth(in_file = raw_mov2_dir,
                        sigma_surf = sigma,
                        sigma_vol = sigma,
                        direction = 'COLUMN',
                        right_surf = surf_r_dir,
                        left_surf = surf_l_dir,
                        out_file = ss_mov2_dir)
    os.system(smooth.cmdline)
    smooth = CiftiSmooth(in_file = raw_mov3_dir,
                        sigma_surf = sigma,
                        sigma_vol = sigma,
                        direction = 'COLUMN',
                        right_surf = surf_r_dir,
                        left_surf = surf_l_dir,
                        out_file = ss_mov3_dir)
    os.system(smooth.cmdline)
    smooth = CiftiSmooth(in_file = raw_mov4_dir,
                        sigma_surf = sigma,
                        sigma_vol = sigma,
                        direction = 'COLUMN',
                        right_surf = surf_r_dir,
                        left_surf = surf_l_dir,
                        out_file = ss_mov4_dir)
    os.system(smooth.cmdline)

    # task
    smooth = CiftiSmooth(in_file = raw_tsk1_dir,
                        sigma_surf = sigma,
                        sigma_vol = sigma,
                        direction = 'COLUMN',
                        right_surf = surf_r_dir,
                        left_surf = surf_l_dir,
                        out_file = ss_tsk1_dir)
    os.system(smooth.cmdline)
    smooth = CiftiSmooth(in_file = raw_tsk2_dir,
                        sigma_surf = sigma,
                        sigma_vol = sigma,
                        direction = 'COLUMN',
                        right_surf = surf_r_dir,
                        left_surf = surf_l_dir,
                        out_file = ss_tsk2_dir)
    os.system(smooth.cmdline)
    smooth = CiftiSmooth(in_file = raw_tsk3_dir,
                        sigma_surf = sigma,
                        sigma_vol = sigma,
                        direction = 'COLUMN',
                        right_surf = surf_r_dir,
                        left_surf = surf_l_dir,
                        out_file = ss_tsk3_dir)
    os.system(smooth.cmdline)
    smooth = CiftiSmooth(in_file = raw_tsk4_dir,
                        sigma_surf = sigma,
                        sigma_vol = sigma,
                        direction = 'COLUMN',
                        right_surf = surf_r_dir,
                        left_surf = surf_l_dir,
                        out_file = ss_tsk4_dir)
    os.system(smooth.cmdline)
    smooth = CiftiSmooth(in_file = raw_tsk5_dir,
                        sigma_surf = sigma,
                        sigma_vol = sigma,
                        direction = 'COLUMN',
                        right_surf = surf_r_dir,
                        left_surf = surf_l_dir,
                        out_file = ss_tsk5_dir)
    os.system(smooth.cmdline)
    smooth = CiftiSmooth(in_file = raw_tsk6_dir,
                        sigma_surf = sigma,
                        sigma_vol = sigma,
                        direction = 'COLUMN',
                        right_surf = surf_r_dir,
                        left_surf = surf_l_dir,
                        out_file = ss_tsk6_dir)
    os.system(smooth.cmdline)
    print("Successful Spatial Smoothing Process! Data saved in /data_qnap/yifeis/new/processed.")
spat_smoothing(subject)

# ------ Read spatial smoothed data ------ #
ss_mov1_cifti = nib.load(ss_mov1_dir)
ss_mov2_cifti = nib.load(ss_mov2_dir)
ss_mov3_cifti = nib.load(ss_mov3_dir)
ss_mov4_cifti = nib.load(ss_mov4_dir)

ss_rst1_cifti = nib.load(ss_rst1_dir)
ss_rst2_cifti = nib.load(ss_rst2_dir)
ss_rst3_cifti = nib.load(ss_rst3_dir)
ss_rst4_cifti = nib.load(ss_rst4_dir)

ss_tsk1_cifti = nib.load(ss_tsk1_dir)
ss_tsk2_cifti = nib.load(ss_tsk2_dir)
ss_tsk3_cifti = nib.load(ss_tsk3_dir)
ss_tsk4_cifti = nib.load(ss_tsk4_dir)
ss_tsk5_cifti = nib.load(ss_tsk5_dir)
ss_tsk6_cifti = nib.load(ss_tsk6_dir)

r1vol_s, r1left_s, r1right_s = decompose_cifti(ss_rst1_cifti)
r2vol_s, r2left_s, r2right_s = decompose_cifti(ss_rst2_cifti)
r3vol_s, r3left_s, r3right_s = decompose_cifti(ss_rst3_cifti)
r4vol_s, r4left_s, r4right_s = decompose_cifti(ss_rst4_cifti)

m1vol_s, m1left_s, m1right_s = decompose_cifti(ss_mov1_cifti)
m2vol_s, m2left_s, m2right_s = decompose_cifti(ss_mov2_cifti)
m3vol_s, m3left_s, m3right_s = decompose_cifti(ss_mov3_cifti)
m4vol_s, m4left_s, m4right_s = decompose_cifti(ss_mov4_cifti)

t1vol_s, t1left_s, t1right_s = decompose_cifti(ss_tsk1_cifti)
t2vol_s, t2left_s, t2right_s = decompose_cifti(ss_tsk2_cifti)
t3vol_s, t3left_s, t3right_s = decompose_cifti(ss_tsk3_cifti)
t4vol_s, t4left_s, t4right_s = decompose_cifti(ss_tsk4_cifti)
t5vol_s, t5left_s, t5right_s = decompose_cifti(ss_tsk5_cifti)
t6vol_s, t6left_s, t6right_s = decompose_cifti(ss_tsk6_cifti)

# ------ Temporal Bandpass Filter ------ #
# https://nipype.readthedocs.io/en/latest/users/examples/rsfmri_vol_surface_preprocessing_nipy.html
# filter in whole matrix
tf_mov1_dir = '/data_qnap/yifeis/new/processed/'+str(subject)+'/temporal_filtered_tfMRI_MOVIE1_7T_AP_Atlas_MSMAll_hp2000_clean.dtseries.nii'
tf_mov2_dir = '/data_qnap/yifeis/new/processed/'+str(subject)+'/temporal_filtered_tfMRI_MOVIE2_7T_PA_Atlas_MSMAll_hp2000_clean.dtseries.nii'
tf_mov3_dir = '/data_qnap/yifeis/new/processed/'+str(subject)+'/temporal_filtered_tfMRI_MOVIE3_7T_PA_Atlas_MSMAll_hp2000_clean.dtseries.nii'
tf_mov4_dir = '/data_qnap/yifeis/new/processed/'+str(subject)+'/temporal_filtered_tfMRI_MOVIE4_7T_AP_Atlas_MSMAll_hp2000_clean.dtseries.nii'

tf_rst1_dir = '/data_qnap/yifeis/new/processed/'+str(subject)+'/temporal_filtered_rfMRI_REST1_7T_PA_Atlas_MSMAll_hp2000_clean.dtseries.nii'
tf_rst2_dir = '/data_qnap/yifeis/new/processed/'+str(subject)+'/temporal_filtered_rfMRI_REST2_7T_AP_Atlas_MSMAll_hp2000_clean.dtseries.nii'
tf_rst3_dir = '/data_qnap/yifeis/new/processed/'+str(subject)+'/temporal_filtered_rfMRI_REST3_7T_PA_Atlas_MSMAll_hp2000_clean.dtseries.nii'
tf_rst4_dir = '/data_qnap/yifeis/new/processed/'+str(subject)+'/temporal_filtered_rfMRI_REST4_7T_AP_Atlas_MSMAll_hp2000_clean.dtseries.nii'

tf_tsk1_dir = '/data_qnap/yifeis/new/processed/'+str(subject)+'/temporal_filtered_tfMRI_RETBAR1_7T_AP_Atlas_MSMAll_hp2000_clean.dtseries.nii'
tf_tsk2_dir = '/data_qnap/yifeis/new/processed/'+str(subject)+'/temporal_filtered_tfMRI_RETBAR2_7T_PA_Atlas_MSMAll_hp2000_clean.dtseries.nii'
tf_tsk3_dir = '/data_qnap/yifeis/new/processed/'+str(subject)+'/temporal_filtered_tfMRI_RETCCW_7T_AP_Atlas_MSMAll_hp2000_clean.dtseries.nii'
tf_tsk4_dir = '/data_qnap/yifeis/new/processed/'+str(subject)+'/temporal_filtered_tfMRI_RETCON_7T_PA_Atlas_MSMAll_hp2000_clean.dtseries.nii'
tf_tsk5_dir = '/data_qnap/yifeis/new/processed/'+str(subject)+'/temporal_filtered_tfMRI_RETCW_7T_PA_Atlas_MSMAll_hp2000_clean.dtseries.nii'
tf_tsk6_dir = '/data_qnap/yifeis/new/processed/'+str(subject)+'/temporal_filtered_tfMRI_RETEXP_7T_AP_Atlas_MSMAll_hp2000_clean.dtseries.nii'

def temp_filtering(subject):
    # spatial smoothed data dir
    ss_mov1_dir = '/data_qnap/yifeis/new/processed/'+str(subject)+'/spatial_smoothed_tfMRI_MOVIE1_7T_AP_Atlas_MSMAll_hp2000_clean.dtseries.nii'
    ss_mov2_dir = '/data_qnap/yifeis/new/processed/'+str(subject)+'/spatial_smoothed_tfMRI_MOVIE2_7T_PA_Atlas_MSMAll_hp2000_clean.dtseries.nii'
    ss_mov3_dir = '/data_qnap/yifeis/new/processed/'+str(subject)+'/spatial_smoothed_tfMRI_MOVIE3_7T_PA_Atlas_MSMAll_hp2000_clean.dtseries.nii'
    ss_mov4_dir = '/data_qnap/yifeis/new/processed/'+str(subject)+'/spatial_smoothed_tfMRI_MOVIE4_7T_AP_Atlas_MSMAll_hp2000_clean.dtseries.nii'

    ss_rst1_dir = '/data_qnap/yifeis/new/processed/'+str(subject)+'/spatial_smoothed_rfMRI_REST1_7T_PA_Atlas_MSMAll_hp2000_clean.dtseries.nii'
    ss_rst2_dir = '/data_qnap/yifeis/new/processed/'+str(subject)+'/spatial_smoothed_rfMRI_REST2_7T_AP_Atlas_MSMAll_hp2000_clean.dtseries.nii'
    ss_rst3_dir = '/data_qnap/yifeis/new/processed/'+str(subject)+'/spatial_smoothed_rfMRI_REST3_7T_PA_Atlas_MSMAll_hp2000_clean.dtseries.nii'
    ss_rst4_dir = '/data_qnap/yifeis/new/processed/'+str(subject)+'/spatial_smoothed_rfMRI_REST4_7T_AP_Atlas_MSMAll_hp2000_clean.dtseries.nii'

    ss_tsk1_dir = '/data_qnap/yifeis/new/processed/'+str(subject)+'/spatial_smoothed_tfMRI_RETBAR1_7T_AP_Atlas_MSMAll_hp2000_clean.dtseries.nii'
    ss_tsk2_dir = '/data_qnap/yifeis/new/processed/'+str(subject)+'/spatial_smoothed_tfMRI_RETBAR2_7T_PA_Atlas_MSMAll_hp2000_clean.dtseries.nii'
    ss_tsk3_dir = '/data_qnap/yifeis/new/processed/'+str(subject)+'/spatial_smoothed_tfMRI_RETCCW_7T_AP_Atlas_MSMAll_hp2000_clean.dtseries.nii'
    ss_tsk4_dir = '/data_qnap/yifeis/new/processed/'+str(subject)+'/spatial_smoothed_tfMRI_RETCON_7T_PA_Atlas_MSMAll_hp2000_clean.dtseries.nii'
    ss_tsk5_dir = '/data_qnap/yifeis/new/processed/'+str(subject)+'/spatial_smoothed_tfMRI_RETCW_7T_PA_Atlas_MSMAll_hp2000_clean.dtseries.nii'
    ss_tsk6_dir = '/data_qnap/yifeis/new/processed/'+str(subject)+'/spatial_smoothed_tfMRI_RETEXP_7T_AP_Atlas_MSMAll_hp2000_clean.dtseries.nii'
    # temporal filtered data dir
    tf_mov1_dir = '/data_qnap/yifeis/new/processed/'+str(subject)+'/temporal_filtered_tfMRI_MOVIE1_7T_AP_Atlas_MSMAll_hp2000_clean.dtseries.nii'
    tf_mov2_dir = '/data_qnap/yifeis/new/processed/'+str(subject)+'/temporal_filtered_tfMRI_MOVIE2_7T_PA_Atlas_MSMAll_hp2000_clean.dtseries.nii'
    tf_mov3_dir = '/data_qnap/yifeis/new/processed/'+str(subject)+'/temporal_filtered_tfMRI_MOVIE3_7T_PA_Atlas_MSMAll_hp2000_clean.dtseries.nii'
    tf_mov4_dir = '/data_qnap/yifeis/new/processed/'+str(subject)+'/temporal_filtered_tfMRI_MOVIE4_7T_AP_Atlas_MSMAll_hp2000_clean.dtseries.nii'

    tf_rst1_dir = '/data_qnap/yifeis/new/processed/'+str(subject)+'/temporal_filtered_rfMRI_REST1_7T_PA_Atlas_MSMAll_hp2000_clean.dtseries.nii'
    tf_rst2_dir = '/data_qnap/yifeis/new/processed/'+str(subject)+'/temporal_filtered_rfMRI_REST2_7T_AP_Atlas_MSMAll_hp2000_clean.dtseries.nii'
    tf_rst3_dir = '/data_qnap/yifeis/new/processed/'+str(subject)+'/temporal_filtered_rfMRI_REST3_7T_PA_Atlas_MSMAll_hp2000_clean.dtseries.nii'
    tf_rst4_dir = '/data_qnap/yifeis/new/processed/'+str(subject)+'/temporal_filtered_rfMRI_REST4_7T_AP_Atlas_MSMAll_hp2000_clean.dtseries.nii'

    tf_tsk1_dir = '/data_qnap/yifeis/new/processed/'+str(subject)+'/temporal_filtered_tfMRI_RETBAR1_7T_AP_Atlas_MSMAll_hp2000_clean.dtseries.nii'
    tf_tsk2_dir = '/data_qnap/yifeis/new/processed/'+str(subject)+'/temporal_filtered_tfMRI_RETBAR2_7T_PA_Atlas_MSMAll_hp2000_clean.dtseries.nii'
    tf_tsk3_dir = '/data_qnap/yifeis/new/processed/'+str(subject)+'/temporal_filtered_tfMRI_RETCCW_7T_AP_Atlas_MSMAll_hp2000_clean.dtseries.nii'
    tf_tsk4_dir = '/data_qnap/yifeis/new/processed/'+str(subject)+'/temporal_filtered_tfMRI_RETCON_7T_PA_Atlas_MSMAll_hp2000_clean.dtseries.nii'
    tf_tsk5_dir = '/data_qnap/yifeis/new/processed/'+str(subject)+'/temporal_filtered_tfMRI_RETCW_7T_PA_Atlas_MSMAll_hp2000_clean.dtseries.nii'
    tf_tsk6_dir = '/data_qnap/yifeis/new/processed/'+str(subject)+'/temporal_filtered_tfMRI_RETEXP_7T_AP_Atlas_MSMAll_hp2000_clean.dtseries.nii'
    print("------ Is Doing Temporal Filtering ------")
    m1_mtx = bandpass_filter(ss_mov1_cifti, 1)
    m2_mtx = bandpass_filter(ss_mov2_cifti, 1)
    m3_mtx = bandpass_filter(ss_mov3_cifti, 1)
    m4_mtx = bandpass_filter(ss_mov4_cifti, 1)

    r1_mtx = bandpass_filter(ss_rst1_cifti, 1)
    r2_mtx = bandpass_filter(ss_rst2_cifti, 1)
    r3_mtx = bandpass_filter(ss_rst3_cifti, 1)
    r4_mtx = bandpass_filter(ss_rst4_cifti, 1)

    t1_mtx = bandpass_filter(ss_tsk1_cifti, 1)
    t2_mtx = bandpass_filter(ss_tsk2_cifti, 1)
    t3_mtx = bandpass_filter(ss_tsk3_cifti, 1)
    t4_mtx = bandpass_filter(ss_tsk4_cifti, 1)
    t5_mtx = bandpass_filter(ss_tsk5_cifti, 1)
    t6_mtx = bandpass_filter(ss_tsk6_cifti, 1)
    # save to cifti file
    nib.save(nib.Cifti2Image(m1_mtx, ss_mov1_cifti.header, ss_mov1_cifti.nifti_header, ss_mov1_cifti.extra, ss_mov1_cifti.file_map), tf_mov1_dir)
    nib.save(nib.Cifti2Image(m2_mtx, ss_mov2_cifti.header, ss_mov2_cifti.nifti_header, ss_mov2_cifti.extra, ss_mov2_cifti.file_map), tf_mov2_dir)
    nib.save(nib.Cifti2Image(m3_mtx, ss_mov3_cifti.header, ss_mov3_cifti.nifti_header, ss_mov3_cifti.extra, ss_mov3_cifti.file_map), tf_mov3_dir)
    nib.save(nib.Cifti2Image(m4_mtx, ss_mov4_cifti.header, ss_mov4_cifti.nifti_header, ss_mov4_cifti.extra, ss_mov4_cifti.file_map), tf_mov4_dir)

    nib.save(nib.Cifti2Image(r1_mtx, ss_rst1_cifti.header, ss_rst1_cifti.nifti_header, ss_rst1_cifti.extra, ss_rst1_cifti.file_map), tf_rst1_dir)
    nib.save(nib.Cifti2Image(r2_mtx, ss_rst2_cifti.header, ss_rst2_cifti.nifti_header, ss_rst2_cifti.extra, ss_rst2_cifti.file_map), tf_rst2_dir)
    nib.save(nib.Cifti2Image(r3_mtx, ss_rst3_cifti.header, ss_rst3_cifti.nifti_header, ss_rst3_cifti.extra, ss_rst3_cifti.file_map), tf_rst3_dir)
    nib.save(nib.Cifti2Image(r4_mtx, ss_rst4_cifti.header, ss_rst4_cifti.nifti_header, ss_rst4_cifti.extra, ss_rst4_cifti.file_map), tf_rst4_dir)

    nib.save(nib.Cifti2Image(t1_mtx, ss_tsk1_cifti.header, ss_tsk1_cifti.nifti_header, ss_tsk1_cifti.extra, ss_tsk1_cifti.file_map), tf_tsk1_dir)
    nib.save(nib.Cifti2Image(t2_mtx, ss_tsk2_cifti.header, ss_tsk2_cifti.nifti_header, ss_tsk2_cifti.extra, ss_tsk2_cifti.file_map), tf_tsk2_dir)
    nib.save(nib.Cifti2Image(t3_mtx, ss_tsk3_cifti.header, ss_tsk3_cifti.nifti_header, ss_tsk3_cifti.extra, ss_tsk3_cifti.file_map), tf_tsk3_dir)
    nib.save(nib.Cifti2Image(t4_mtx, ss_tsk4_cifti.header, ss_tsk4_cifti.nifti_header, ss_tsk4_cifti.extra, ss_tsk4_cifti.file_map), tf_tsk4_dir)
    nib.save(nib.Cifti2Image(t5_mtx, ss_tsk5_cifti.header, ss_tsk5_cifti.nifti_header, ss_tsk5_cifti.extra, ss_tsk5_cifti.file_map), tf_tsk5_dir)
    nib.save(nib.Cifti2Image(t6_mtx, ss_tsk6_cifti.header, ss_tsk6_cifti.nifti_header, ss_tsk6_cifti.extra, ss_tsk6_cifti.file_map), tf_tsk6_dir)
    print("------ Temporal Filtering Done! ------")
temp_filtering(subject)

tf_mov1_cifti = nib.load(tf_mov1_dir)
tf_mov2_cifti = nib.load(tf_mov2_dir)
tf_mov3_cifti = nib.load(tf_mov3_dir)
tf_mov4_cifti = nib.load(tf_mov4_dir)

tf_rst1_cifti = nib.load(tf_rst1_dir)
tf_rst2_cifti = nib.load(tf_rst2_dir)
tf_rst3_cifti = nib.load(tf_rst3_dir)
tf_rst4_cifti = nib.load(tf_rst4_dir)

tf_tsk1_cifti = nib.load(tf_tsk1_dir)
tf_tsk2_cifti = nib.load(tf_tsk2_dir)
tf_tsk3_cifti = nib.load(tf_tsk3_dir)
tf_tsk4_cifti = nib.load(tf_tsk4_dir)
tf_tsk5_cifti = nib.load(tf_tsk5_dir)
tf_tsk6_cifti = nib.load(tf_tsk6_dir)

m1_mtx = tf_mov1_cifti.get_fdata(dtype=np.float32)
m2_mtx = tf_mov2_cifti.get_fdata(dtype=np.float32)
m3_mtx = tf_mov3_cifti.get_fdata(dtype=np.float32)
m4_mtx = tf_mov4_cifti.get_fdata(dtype=np.float32)

r1_mtx = tf_rst1_cifti.get_fdata(dtype=np.float32)
r2_mtx = tf_rst2_cifti.get_fdata(dtype=np.float32)
r3_mtx = tf_rst3_cifti.get_fdata(dtype=np.float32)
r4_mtx = tf_rst4_cifti.get_fdata(dtype=np.float32)

t1_mtx = tf_tsk1_cifti.get_fdata(dtype=np.float32)
t2_mtx = tf_tsk2_cifti.get_fdata(dtype=np.float32)
t3_mtx = tf_tsk3_cifti.get_fdata(dtype=np.float32)
t4_mtx = tf_tsk4_cifti.get_fdata(dtype=np.float32)
t5_mtx = tf_tsk5_cifti.get_fdata(dtype=np.float32)
t6_mtx = tf_tsk6_cifti.get_fdata(dtype=np.float32)

r1left_s_t  = surf_data_from_cifti(r1_mtx, tf_rst1_cifti.header.get_axis(1), "CIFTI_STRUCTURE_CORTEX_LEFT")
r1right_s_t = surf_data_from_cifti(r1_mtx, tf_rst1_cifti.header.get_axis(1), "CIFTI_STRUCTURE_CORTEX_RIGHT")
r2left_s_t  = surf_data_from_cifti(r2_mtx, tf_rst2_cifti.header.get_axis(1), "CIFTI_STRUCTURE_CORTEX_LEFT")
r2right_s_t = surf_data_from_cifti(r2_mtx, tf_rst2_cifti.header.get_axis(1), "CIFTI_STRUCTURE_CORTEX_RIGHT")
r3left_s_t  = surf_data_from_cifti(r3_mtx, tf_rst3_cifti.header.get_axis(1), "CIFTI_STRUCTURE_CORTEX_LEFT")
r3right_s_t = surf_data_from_cifti(r3_mtx, tf_rst3_cifti.header.get_axis(1), "CIFTI_STRUCTURE_CORTEX_RIGHT")
r4left_s_t  = surf_data_from_cifti(r4_mtx, tf_rst4_cifti.header.get_axis(1), "CIFTI_STRUCTURE_CORTEX_LEFT")
r4right_s_t = surf_data_from_cifti(r4_mtx, tf_rst4_cifti.header.get_axis(1), "CIFTI_STRUCTURE_CORTEX_RIGHT")

m1left_s_t  = surf_data_from_cifti(m1_mtx, tf_mov1_cifti.header.get_axis(1), "CIFTI_STRUCTURE_CORTEX_LEFT")
m1right_s_t = surf_data_from_cifti(m1_mtx, tf_mov1_cifti.header.get_axis(1), "CIFTI_STRUCTURE_CORTEX_RIGHT")
m2left_s_t  = surf_data_from_cifti(m2_mtx, tf_mov2_cifti.header.get_axis(1), "CIFTI_STRUCTURE_CORTEX_LEFT")
m2right_s_t = surf_data_from_cifti(m2_mtx, tf_mov2_cifti.header.get_axis(1), "CIFTI_STRUCTURE_CORTEX_RIGHT")
m3left_s_t  = surf_data_from_cifti(m3_mtx, tf_mov3_cifti.header.get_axis(1), "CIFTI_STRUCTURE_CORTEX_LEFT")
m3right_s_t = surf_data_from_cifti(m3_mtx, tf_mov3_cifti.header.get_axis(1), "CIFTI_STRUCTURE_CORTEX_RIGHT")
m4left_s_t  = surf_data_from_cifti(m4_mtx, tf_mov4_cifti.header.get_axis(1), "CIFTI_STRUCTURE_CORTEX_LEFT")
m4right_s_t = surf_data_from_cifti(m4_mtx, tf_mov4_cifti.header.get_axis(1), "CIFTI_STRUCTURE_CORTEX_RIGHT")

t1left_s_t  = surf_data_from_cifti(t1_mtx, tf_tsk1_cifti.header.get_axis(1), "CIFTI_STRUCTURE_CORTEX_LEFT")
t1right_s_t = surf_data_from_cifti(t1_mtx, tf_tsk1_cifti.header.get_axis(1), "CIFTI_STRUCTURE_CORTEX_RIGHT")
t2left_s_t  = surf_data_from_cifti(t2_mtx, tf_tsk2_cifti.header.get_axis(1), "CIFTI_STRUCTURE_CORTEX_LEFT")
t2right_s_t = surf_data_from_cifti(t2_mtx, tf_tsk2_cifti.header.get_axis(1), "CIFTI_STRUCTURE_CORTEX_RIGHT")
t3left_s_t  = surf_data_from_cifti(t3_mtx, tf_tsk3_cifti.header.get_axis(1), "CIFTI_STRUCTURE_CORTEX_LEFT")
t3right_s_t = surf_data_from_cifti(t3_mtx, tf_tsk3_cifti.header.get_axis(1), "CIFTI_STRUCTURE_CORTEX_RIGHT")
t4left_s_t  = surf_data_from_cifti(t4_mtx, tf_tsk4_cifti.header.get_axis(1), "CIFTI_STRUCTURE_CORTEX_LEFT")
t4right_s_t = surf_data_from_cifti(t4_mtx, tf_tsk4_cifti.header.get_axis(1), "CIFTI_STRUCTURE_CORTEX_RIGHT")
t5left_s_t  = surf_data_from_cifti(t5_mtx, tf_tsk5_cifti.header.get_axis(1), "CIFTI_STRUCTURE_CORTEX_LEFT")
t5right_s_t = surf_data_from_cifti(t5_mtx, tf_tsk5_cifti.header.get_axis(1), "CIFTI_STRUCTURE_CORTEX_RIGHT")
t6left_s_t  = surf_data_from_cifti(t6_mtx, tf_tsk6_cifti.header.get_axis(1), "CIFTI_STRUCTURE_CORTEX_LEFT")
t6right_s_t = surf_data_from_cifti(t6_mtx, tf_tsk6_cifti.header.get_axis(1), "CIFTI_STRUCTURE_CORTEX_RIGHT")

# ------ Plot data ------ #
def save_plots():
    os.mkdir('/data_qnap/yifeis/new_plots/'+str(subject))
    os.mkdir('/data_qnap/yifeis/new_plots/'+str(subject)+'/movie')
    os.mkdir('/data_qnap/yifeis/new_plots/'+str(subject)+'/movie/movie1')
    os.mkdir('/data_qnap/yifeis/new_plots/'+str(subject)+'/movie/movie2')
    os.mkdir('/data_qnap/yifeis/new_plots/'+str(subject)+'/movie/movie3')
    os.mkdir('/data_qnap/yifeis/new_plots/'+str(subject)+'/movie/movie4')
    os.mkdir('/data_qnap/yifeis/new_plots/'+str(subject)+'/rest')
    os.mkdir('/data_qnap/yifeis/new_plots/'+str(subject)+'/rest/rest1')
    os.mkdir('/data_qnap/yifeis/new_plots/'+str(subject)+'/rest/rest2')
    os.mkdir('/data_qnap/yifeis/new_plots/'+str(subject)+'/rest/rest3')
    os.mkdir('/data_qnap/yifeis/new_plots/'+str(subject)+'/rest/rest4')
    os.mkdir('/data_qnap/yifeis/new_plots/'+str(subject)+'/task')
    os.mkdir('/data_qnap/yifeis/new_plots/'+str(subject)+'/task/retbar1')
    os.mkdir('/data_qnap/yifeis/new_plots/'+str(subject)+'/task/retbar2')
    os.mkdir('/data_qnap/yifeis/new_plots/'+str(subject)+'/task/retccw')
    os.mkdir('/data_qnap/yifeis/new_plots/'+str(subject)+'/task/retcon')
    os.mkdir('/data_qnap/yifeis/new_plots/'+str(subject)+'/task/retcw')
    os.mkdir('/data_qnap/yifeis/new_plots/'+str(subject)+'/task/retexp')
    print("------Surface Plots Saved in /data_qnap/yifeis/plots------")
    # movie
    nlp.view_surf(hcp.mesh.inflated_left, np.transpose(m1left)[0], bg_map=hcp.mesh.sulc_left).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/movie/movie1/raw_left_cortex.html')
    nlp.view_surf(hcp.mesh.inflated_right, np.transpose(m1right)[0], bg_map=hcp.mesh.sulc_right).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/movie/movie1/raw_right_cortex.html')
    nlp.view_surf(hcp.mesh.inflated_left, np.transpose(m1left_s)[0], bg_map=hcp.mesh.sulc_left).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/movie/movie1/smoothed_left_cortex.html')
    nlp.view_surf(hcp.mesh.inflated_right, np.transpose(m1right_s)[0], bg_map=hcp.mesh.sulc_right).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/movie/movie1/smoothed_right_cortex.html')
    nlp.view_surf(hcp.mesh.inflated_left, np.transpose(m1left_s_t)[0], bg_map=hcp.mesh.sulc_left).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/movie/movie1/preprocessed_left_cortex.html')
    nlp.view_surf(hcp.mesh.inflated_right, np.transpose(m1right_s_t)[0], bg_map=hcp.mesh.sulc_right).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/movie/movie1/preprocessed_right_cortex.html')
    nlp.view_surf(hcp.mesh.inflated, hcp.cortex_data(m1_mtx[0]), bg_map=hcp.mesh.sulc).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/movie/movie1/preprocessed_cortex.html')

    nlp.view_surf(hcp.mesh.inflated_left, np.transpose(m2left)[0], bg_map=hcp.mesh.sulc_left).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/movie/movie2/raw_left_cortex.html')
    nlp.view_surf(hcp.mesh.inflated_right, np.transpose(m2right)[0], bg_map=hcp.mesh.sulc_right).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/movie/movie2/raw_right_cortex.html')
    nlp.view_surf(hcp.mesh.inflated_left, np.transpose(m2left_s)[0], bg_map=hcp.mesh.sulc_left).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/movie/movie2/smoothed_left_cortex.html')
    nlp.view_surf(hcp.mesh.inflated_right, np.transpose(m2right_s)[0], bg_map=hcp.mesh.sulc_right).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/movie/movie2/smoothed_right_cortex.html')
    nlp.view_surf(hcp.mesh.inflated_left, np.transpose(m2left_s_t)[0], bg_map=hcp.mesh.sulc_left).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/movie/movie2/preprocessed_left_cortex.html')
    nlp.view_surf(hcp.mesh.inflated_right, np.transpose(m2right_s_t)[0], bg_map=hcp.mesh.sulc_right).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/movie/movie2/preprocessed_right_cortex.html')
    nlp.view_surf(hcp.mesh.inflated, hcp.cortex_data(m2_mtx[0]), bg_map=hcp.mesh.sulc).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/movie/movie2/preprocessed_cortex.html')

    nlp.view_surf(hcp.mesh.inflated_left, np.transpose(m3left)[0], bg_map=hcp.mesh.sulc_left).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/movie/movie3/raw_left_cortex.html')
    nlp.view_surf(hcp.mesh.inflated_right, np.transpose(m3right)[0], bg_map=hcp.mesh.sulc_right).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/movie/movie3/raw_right_cortex.html')
    nlp.view_surf(hcp.mesh.inflated_left, np.transpose(m3left_s)[0], bg_map=hcp.mesh.sulc_left).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/movie/movie3/smoothed_left_cortex.html')
    nlp.view_surf(hcp.mesh.inflated_right, np.transpose(m3right_s)[0], bg_map=hcp.mesh.sulc_right).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/movie/movie3/smoothed_right_cortex.html')
    nlp.view_surf(hcp.mesh.inflated_left, np.transpose(m3left_s_t)[0], bg_map=hcp.mesh.sulc_left).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/movie/movie3/preprocessed_left_cortex.html')
    nlp.view_surf(hcp.mesh.inflated_right, np.transpose(m3right_s_t)[0], bg_map=hcp.mesh.sulc_right).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/movie/movie3/preprocessed_right_cortex.html')
    nlp.view_surf(hcp.mesh.inflated, hcp.cortex_data(m3_mtx[0]), bg_map=hcp.mesh.sulc).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/movie/movie3/preprocessed_cortex.html')

    nlp.view_surf(hcp.mesh.inflated_left, np.transpose(m4left)[0], bg_map=hcp.mesh.sulc_left).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/movie/movie4/raw_left_cortex.html')
    nlp.view_surf(hcp.mesh.inflated_right, np.transpose(m4right)[0], bg_map=hcp.mesh.sulc_right).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/movie/movie4/raw_right_cortex.html')
    nlp.view_surf(hcp.mesh.inflated_left, np.transpose(m4left_s)[0], bg_map=hcp.mesh.sulc_left).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/movie/movie4/smoothed_left_cortex.html')
    nlp.view_surf(hcp.mesh.inflated_right, np.transpose(m4right_s)[0], bg_map=hcp.mesh.sulc_right).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/movie/movie4/smoothed_right_cortex.html')
    nlp.view_surf(hcp.mesh.inflated_left, np.transpose(m4left_s_t)[0], bg_map=hcp.mesh.sulc_left).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/movie/movie4/preprocessed_left_cortex.html')
    nlp.view_surf(hcp.mesh.inflated_right, np.transpose(m4right_s_t)[0], bg_map=hcp.mesh.sulc_right).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/movie/movie4/preprocessed_right_cortex.html')
    nlp.view_surf(hcp.mesh.inflated, hcp.cortex_data(m4_mtx[0]), bg_map=hcp.mesh.sulc).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/movie/movie4/preprocessed_cortex.html')

    # rest
    nlp.view_surf(hcp.mesh.inflated_left, np.transpose(r1left)[0], bg_map=hcp.mesh.sulc_left).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/rest/rest1/raw_left_cortex.html')
    nlp.view_surf(hcp.mesh.inflated_right, np.transpose(r1right)[0], bg_map=hcp.mesh.sulc_right).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/rest/rest1/raw_right_cortex.html')
    nlp.view_surf(hcp.mesh.inflated_left, np.transpose(r1left_s)[0], bg_map=hcp.mesh.sulc_left).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/rest/rest1/smoothed_left_cortex.html')
    nlp.view_surf(hcp.mesh.inflated_right, np.transpose(r1right_s)[0], bg_map=hcp.mesh.sulc_right).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/rest/rest1/smoothed_right_cortex.html')
    nlp.view_surf(hcp.mesh.inflated_left, np.transpose(r1left_s_t)[0], bg_map=hcp.mesh.sulc_left).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/rest/rest1/preprocessed_left_cortex.html')
    nlp.view_surf(hcp.mesh.inflated_right, np.transpose(r1right_s_t)[0], bg_map=hcp.mesh.sulc_right).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/rest/rest1/preprocessed_right_cortex.html')
    nlp.view_surf(hcp.mesh.inflated, hcp.cortex_data(r1_mtx[0]), bg_map=hcp.mesh.sulc).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/rest/rest1/preprocessed_cortex.html')

    nlp.view_surf(hcp.mesh.inflated_left, np.transpose(r2left)[0], bg_map=hcp.mesh.sulc_left).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/rest/rest2/raw_left_cortex.html')
    nlp.view_surf(hcp.mesh.inflated_right, np.transpose(r2right)[0], bg_map=hcp.mesh.sulc_right).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/rest/rest2/raw_right_cortex.html')
    nlp.view_surf(hcp.mesh.inflated_left, np.transpose(r2left_s)[0], bg_map=hcp.mesh.sulc_left).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/rest/rest2/smoothed_left_cortex.html')
    nlp.view_surf(hcp.mesh.inflated_right, np.transpose(r2right_s)[0], bg_map=hcp.mesh.sulc_right).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/rest/rest2/smoothed_right_cortex.html')
    nlp.view_surf(hcp.mesh.inflated_left, np.transpose(r2left_s_t)[0], bg_map=hcp.mesh.sulc_left).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/rest/rest2/preprocessed_left_cortex.html')
    nlp.view_surf(hcp.mesh.inflated_right, np.transpose(r2right_s_t)[0], bg_map=hcp.mesh.sulc_right).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/rest/rest2/preprocessed_right_cortex.html')
    nlp.view_surf(hcp.mesh.inflated, hcp.cortex_data(r2_mtx[0]), bg_map=hcp.mesh.sulc).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/rest/rest2/preprocessed_cortex.html')

    nlp.view_surf(hcp.mesh.inflated_left, np.transpose(r3left)[0], bg_map=hcp.mesh.sulc_left).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/rest/rest3/raw_left_cortex.html')
    nlp.view_surf(hcp.mesh.inflated_right, np.transpose(r3right)[0], bg_map=hcp.mesh.sulc_right).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/rest/rest3/raw_right_cortex.html')
    nlp.view_surf(hcp.mesh.inflated_left, np.transpose(r3left_s)[0], bg_map=hcp.mesh.sulc_left).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/rest/rest3/smoothed_left_cortex.html')
    nlp.view_surf(hcp.mesh.inflated_right, np.transpose(r3right_s)[0], bg_map=hcp.mesh.sulc_right).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/rest/rest3/smoothed_right_cortex.html')
    nlp.view_surf(hcp.mesh.inflated_left, np.transpose(r3left_s_t)[0], bg_map=hcp.mesh.sulc_left).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/rest/rest3/preprocessed_left_cortex.html')
    nlp.view_surf(hcp.mesh.inflated_right, np.transpose(r3right_s_t)[0], bg_map=hcp.mesh.sulc_right).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/rest/rest3/preprocessed_right_cortex.html')
    nlp.view_surf(hcp.mesh.inflated, hcp.cortex_data(r3_mtx[0]), bg_map=hcp.mesh.sulc).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/rest/rest3/preprocessed_cortex.html')

    nlp.view_surf(hcp.mesh.inflated_left, np.transpose(r4left)[0], bg_map=hcp.mesh.sulc_left).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/rest/rest4/raw_left_cortex.html')
    nlp.view_surf(hcp.mesh.inflated_right, np.transpose(r4right)[0], bg_map=hcp.mesh.sulc_right).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/rest/rest4/raw_right_cortex.html')
    nlp.view_surf(hcp.mesh.inflated_left, np.transpose(r4left_s)[0], bg_map=hcp.mesh.sulc_left).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/rest/rest4/smoothed_left_cortex.html')
    nlp.view_surf(hcp.mesh.inflated_right, np.transpose(r4right_s)[0], bg_map=hcp.mesh.sulc_right).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/rest/rest4/smoothed_right_cortex.html')
    nlp.view_surf(hcp.mesh.inflated_left, np.transpose(r4left_s_t)[0], bg_map=hcp.mesh.sulc_left).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/rest/rest4/preprocessed_left_cortex.html')
    nlp.view_surf(hcp.mesh.inflated_right, np.transpose(r4right_s_t)[0], bg_map=hcp.mesh.sulc_right).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/rest/rest4/preprocessed_right_cortex.html')
    nlp.view_surf(hcp.mesh.inflated, hcp.cortex_data(r4_mtx[0]), bg_map=hcp.mesh.sulc).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/rest/rest4/preprocessed_cortex.html')
    # task
    nlp.view_surf(hcp.mesh.inflated_left, np.transpose(t1left)[0], bg_map=hcp.mesh.sulc_left).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/task/retbar1/raw_left_cortex.html')
    nlp.view_surf(hcp.mesh.inflated_right, np.transpose(t1right)[0], bg_map=hcp.mesh.sulc_right).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/task/retbar1/raw_right_cortex.html')
    nlp.view_surf(hcp.mesh.inflated_left, np.transpose(t1left_s)[0], bg_map=hcp.mesh.sulc_left).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/task/retbar1/smoothed_left_cortex.html')
    nlp.view_surf(hcp.mesh.inflated_right, np.transpose(t1right_s)[0], bg_map=hcp.mesh.sulc_right).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/task/retbar1/smoothed_right_cortex.html')
    nlp.view_surf(hcp.mesh.inflated_left, np.transpose(t1left_s_t)[0], bg_map=hcp.mesh.sulc_left).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/task/retbar1/preprocessed_left_cortex.html')
    nlp.view_surf(hcp.mesh.inflated_right, np.transpose(t1right_s_t)[0], bg_map=hcp.mesh.sulc_right).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/task/retbar1/preprocessed_right_cortex.html')
    nlp.view_surf(hcp.mesh.inflated, hcp.cortex_data(t1_mtx[0]), bg_map=hcp.mesh.sulc).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/task/retbar1/preprocessed_cortex.html')

    nlp.view_surf(hcp.mesh.inflated_left, np.transpose(t2left)[0], bg_map=hcp.mesh.sulc_left).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/task/retbar2/raw_left_cortex.html')
    nlp.view_surf(hcp.mesh.inflated_right, np.transpose(t2right)[0], bg_map=hcp.mesh.sulc_right).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/task/retbar2/raw_right_cortex.html')
    nlp.view_surf(hcp.mesh.inflated_left, np.transpose(t2left_s)[0], bg_map=hcp.mesh.sulc_left).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/task/retbar2/smoothed_left_cortex.html')
    nlp.view_surf(hcp.mesh.inflated_right, np.transpose(t2right_s)[0], bg_map=hcp.mesh.sulc_right).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/task/retbar2/smoothed_right_cortex.html')
    nlp.view_surf(hcp.mesh.inflated_left, np.transpose(t2left_s_t)[0], bg_map=hcp.mesh.sulc_left).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/task/retbar2/preprocessed_left_cortex.html')
    nlp.view_surf(hcp.mesh.inflated_right, np.transpose(t2right_s_t)[0], bg_map=hcp.mesh.sulc_right).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/task/retbar2/preprocessed_right_cortex.html')
    nlp.view_surf(hcp.mesh.inflated, hcp.cortex_data(t2_mtx[0]), bg_map=hcp.mesh.sulc).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/task/retbar2/preprocessed_cortex.html')

    nlp.view_surf(hcp.mesh.inflated_left, np.transpose(t3left)[0], bg_map=hcp.mesh.sulc_left).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/task/retccw/raw_left_cortex.html')
    nlp.view_surf(hcp.mesh.inflated_right, np.transpose(t3right)[0], bg_map=hcp.mesh.sulc_right).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/task/retccw/raw_right_cortex.html')
    nlp.view_surf(hcp.mesh.inflated_left, np.transpose(t3left_s)[0], bg_map=hcp.mesh.sulc_left).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/task/retccw/smoothed_left_cortex.html')
    nlp.view_surf(hcp.mesh.inflated_right, np.transpose(t3right_s)[0], bg_map=hcp.mesh.sulc_right).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/task/retccw/smoothed_right_cortex.html')
    nlp.view_surf(hcp.mesh.inflated_left, np.transpose(t3left_s_t)[0], bg_map=hcp.mesh.sulc_left).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/task/retccw/preprocessed_left_cortex.html')
    nlp.view_surf(hcp.mesh.inflated_right, np.transpose(t3right_s_t)[0], bg_map=hcp.mesh.sulc_right).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/task/retccw/preprocessed_right_cortex.html')
    nlp.view_surf(hcp.mesh.inflated, hcp.cortex_data(t3_mtx[0]), bg_map=hcp.mesh.sulc).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/task/retccw/preprocessed_cortex.html')

    nlp.view_surf(hcp.mesh.inflated_left, np.transpose(t4left)[0], bg_map=hcp.mesh.sulc_left).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/task/retcon/raw_left_cortex.html')
    nlp.view_surf(hcp.mesh.inflated_right, np.transpose(t4right)[0], bg_map=hcp.mesh.sulc_right).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/task/retcon/raw_right_cortex.html')
    nlp.view_surf(hcp.mesh.inflated_left, np.transpose(t4left_s)[0], bg_map=hcp.mesh.sulc_left).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/task/retcon/smoothed_left_cortex.html')
    nlp.view_surf(hcp.mesh.inflated_right, np.transpose(t4right_s)[0], bg_map=hcp.mesh.sulc_right).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/task/retcon/smoothed_right_cortex.html')
    nlp.view_surf(hcp.mesh.inflated_left, np.transpose(t4left_s_t)[0], bg_map=hcp.mesh.sulc_left).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/task/retcon/preprocessed_left_cortex.html')
    nlp.view_surf(hcp.mesh.inflated_right, np.transpose(t4right_s_t)[0], bg_map=hcp.mesh.sulc_right).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/task/retcon/preprocessed_right_cortex.html')
    nlp.view_surf(hcp.mesh.inflated, hcp.cortex_data(t4_mtx[0]), bg_map=hcp.mesh.sulc).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/task/retcon/preprocessed_cortex.html')

    nlp.view_surf(hcp.mesh.inflated_left, np.transpose(t5left)[0], bg_map=hcp.mesh.sulc_left).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/task/retcw/raw_left_cortex.html')
    nlp.view_surf(hcp.mesh.inflated_right, np.transpose(t5right)[0], bg_map=hcp.mesh.sulc_right).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/task/retcw/raw_right_cortex.html')
    nlp.view_surf(hcp.mesh.inflated_left, np.transpose(t5left_s)[0], bg_map=hcp.mesh.sulc_left).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/task/retcw/smoothed_left_cortex.html')
    nlp.view_surf(hcp.mesh.inflated_right, np.transpose(t5right_s)[0], bg_map=hcp.mesh.sulc_right).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/task/retcw/smoothed_right_cortex.html')
    nlp.view_surf(hcp.mesh.inflated_left, np.transpose(t5left_s_t)[0], bg_map=hcp.mesh.sulc_left).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/task/retcw/preprocessed_left_cortex.html')
    nlp.view_surf(hcp.mesh.inflated_right, np.transpose(t5right_s_t)[0], bg_map=hcp.mesh.sulc_right).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/task/retcw/preprocessed_right_cortex.html')
    nlp.view_surf(hcp.mesh.inflated, hcp.cortex_data(t5_mtx[0]), bg_map=hcp.mesh.sulc).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/task/retcw/preprocessed_cortex.html')

    nlp.view_surf(hcp.mesh.inflated_left, np.transpose(t6left)[0], bg_map=hcp.mesh.sulc_left).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/task/retexp/raw_left_cortex.html')
    nlp.view_surf(hcp.mesh.inflated_right, np.transpose(t6right)[0], bg_map=hcp.mesh.sulc_right).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/task/retexp/raw_right_cortex.html')
    nlp.view_surf(hcp.mesh.inflated_left, np.transpose(t6left_s)[0], bg_map=hcp.mesh.sulc_left).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/task/retexp/smoothed_left_cortex.html')
    nlp.view_surf(hcp.mesh.inflated_right, np.transpose(t6right_s)[0], bg_map=hcp.mesh.sulc_right).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/task/retexp/smoothed_right_cortex.html')
    nlp.view_surf(hcp.mesh.inflated_left, np.transpose(t6left_s_t)[0], bg_map=hcp.mesh.sulc_left).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/task/retexp/preprocessed_left_cortex.html')
    nlp.view_surf(hcp.mesh.inflated_right, np.transpose(t6right_s_t)[0], bg_map=hcp.mesh.sulc_right).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/task/retexp/preprocessed_right_cortex.html')
    nlp.view_surf(hcp.mesh.inflated, hcp.cortex_data(t6_mtx[0]), bg_map=hcp.mesh.sulc).save_as_html('/data_qnap/yifeis/new_plots/'+str(subject)+'/task/retexp/preprocessed_cortex.html')
save_plots()

def show_temp_plots():
    print("------Temporal Plots Pop Up, Close to Continues------")
    # # Task
    fig01 = plt.figure()
    ax01 = fig01.add_subplot(1, 1, 1)
    time_t = np.array(list(range(t1left.shape[1])))
    ax01.plot(time_t, t1left[0] - np.mean(t1left[0]), label = 'raw')
    ax01 = fig01.add_subplot(1, 1, 1)
    ax01.plot(time_t, t1left_s_t[0], label = 'temporal filtered')
    fig01.legend()

    fig02 = plt.figure()
    ax02 = fig02.add_subplot(1, 1, 1)
    time_t = np.array(list(range(t2left.shape[1])))
    ax02.plot(time_t, t2left[0] - np.mean(t2left[0]), label = 'raw')
    ax02 = fig02.add_subplot(1, 1, 1)
    ax02.plot(time_t, t2left_s_t[0], label = 'temporal filtered')
    fig02.legend()

    fig03 = plt.figure()
    ax03 = fig03.add_subplot(1, 1, 1)
    time_t = np.array(list(range(t3left.shape[1])))
    ax03.plot(time_t, t3left[0] - np.mean(t3left[0]), label = 'raw')
    ax03 = fig03.add_subplot(1, 1, 1)
    ax03.plot(time_t, t3left_s_t[0], label = 'temporal filtered')
    fig03.legend()

    fig04 = plt.figure()
    ax04 = fig04.add_subplot(1, 1, 1)
    time_t = np.array(list(range(t4right.shape[1])))
    ax04.plot(time_t, t4right[0] - np.mean(t4right[0]), label = 'raw')
    ax04 = fig04.add_subplot(1, 1, 1)
    ax04.plot(time_t, t4right_s_t[0], label = 'temporal filtered')
    fig04.legend()

    fig05 = plt.figure()
    ax05 = fig05.add_subplot(1, 1, 1)
    time_t = np.array(list(range(t5right.shape[1])))
    ax05.plot(time_t, t5right[0] - np.mean(t5right[0]), label = 'raw')
    ax05 = fig05.add_subplot(1, 1, 1)
    ax05.plot(time_t, t5right_s_t[0], label = 'temporal filtered')
    fig05.legend()

    fig06 = plt.figure()
    ax06 = fig06.add_subplot(1, 1, 1)
    time_t = np.array(list(range(t6right.shape[1])))
    ax06.plot(time_t, t6right[0] - np.mean(t6right[0]), label = 'raw')
    ax06 = fig06.add_subplot(1, 1, 1)
    ax06.plot(time_t, t6right_s_t[0], label = 'temporal filtered')
    fig06.legend()

    # movie
    # fig01 = plt.figure()
    # ax01 = fig01.add_subplot(1, 1, 1)
    # time_m = np.array(list(range(m1left.shape[1])))
    # ax01.plot(time_m, m1left[0] - np.mean(m1left[0]), label = 'raw')
    # ax01 = fig01.add_subplot(1, 1, 1)
    # ax01.plot(time_m, m1left_s_t[0], label = 'temporal filtered')
    # fig01.legend()
    #
    # fig02 = plt.figure()
    # ax02 = fig02.add_subplot(1, 1, 1)
    # time_m = np.array(list(range(m2left.shape[1])))
    # ax02.plot(time_m, m2left[0] - np.mean(m2left[0]), label = 'raw')
    # ax02 = fig02.add_subplot(1, 1, 1)
    # ax02.plot(time_m, m2left_s_t[0], label = 'temporal filtered')
    # fig02.legend()
    #
    # fig03 = plt.figure()
    # ax03 = fig03.add_subplot(1, 1, 1)
    # time_m = np.array(list(range(m3left.shape[1])))
    # ax03.plot(time_m, m3left[0] - np.mean(m3left[0]), label = 'raw')
    # ax03 = fig03.add_subplot(1, 1, 1)
    # ax03.plot(time_m, m3left_s_t[0], label = 'temporal filtered')
    # fig03.legend()
    #
    # fig04 = plt.figure()
    # ax04 = fig04.add_subplot(1, 1, 1)
    # time_m = np.array(list(range(m4left.shape[1])))
    # ax04.plot(time_m, m4left[0] - np.mean(m4left[0]), label = 'raw')
    # ax04 = fig04.add_subplot(1, 1, 1)
    # ax04.plot(time_m, m4left_s_t[0], label = 'temporal filtered')
    # fig04.legend()

    # rest
    # fig01 = plt.figure()
    # ax01 = fig01.add_subplot(1, 1, 1)
    # time_r = np.array(list(range(r1left.shape[1])))
    # ax01.plot(time_r, r1left[0] - np.mean(r1left[0]), label = 'raw')
    # ax01 = fig01.add_subplot(1, 1, 1)
    # ax01.plot(time_r, r1left_s_t[0], label = 'temporal filtered')
    # fig01.legend()
    #
    # fig02 = plt.figure()
    # ax02 = fig02.add_subplot(1, 1, 1)
    # time_r = np.array(list(range(r2left.shape[1])))
    # ax02.plot(time_r, r2left[0] - np.mean(r2left[0]), label = 'raw')
    # ax02 = fig02.add_subplot(1, 1, 1)
    # ax02.plot(time_r, r2left_s_t[0], label = 'temporal filtered')
    # fig02.legend()
    #
    # fig03 = plt.figure()
    # ax03 = fig03.add_subplot(1, 1, 1)
    # time_r = np.array(list(range(r3left.shape[1])))
    # ax03.plot(time_r, r3left[0] - np.mean(r3left[0]), label = 'raw')
    # ax03 = fig03.add_subplot(1, 1, 1)
    # ax03.plot(time_r, r3left_s_t[0], label = 'temporal filtered')
    # fig03.legend()
    #
    # fig04 = plt.figure()
    # ax04 = fig04.add_subplot(1, 1, 1)
    # time_r = np.array(list(range(r4left.shape[1])))
    # ax04.plot(time_r, r4left[0] - np.mean(r4left[0]), label = 'raw')
    # ax04 = fig04.add_subplot(1, 1, 1)
    # ax04.plot(time_r, r4left_s_t[0], label = 'temporal filtered')
    # fig04.legend()
    # plt.show()
# show_temp_plots()

# ------ Parcellation ------ #

# Normalization
m1_mtx = hcp.normalize(m1_mtx)
m2_mtx = hcp.normalize(m2_mtx)
m3_mtx = hcp.normalize(m3_mtx)
m4_mtx = hcp.normalize(m4_mtx)

r1_mtx = hcp.normalize(r1_mtx)
r2_mtx = hcp.normalize(r2_mtx)
r3_mtx = hcp.normalize(r3_mtx)
r4_mtx = hcp.normalize(r4_mtx)

t1_mtx = hcp.normalize(t1_mtx)
t2_mtx = hcp.normalize(t2_mtx)
t3_mtx = hcp.normalize(t3_mtx)
t4_mtx = hcp.normalize(t4_mtx)
t5_mtx = hcp.normalize(t5_mtx)
t6_mtx = hcp.normalize(t6_mtx)

labels = hcp.mmp.labels
label_values = list(labels.values())
## map timeseries to atlas
# result is numpy array
rst1_p = hcp.parcellate(r1_mtx, hcp.mmp)
rst2_p = hcp.parcellate(r2_mtx, hcp.mmp)
rst3_p = hcp.parcellate(r3_mtx, hcp.mmp)
rst4_p = hcp.parcellate(r4_mtx, hcp.mmp)

mov1_p = hcp.parcellate(m1_mtx, hcp.mmp)
mov2_p = hcp.parcellate(m2_mtx, hcp.mmp)
mov3_p = hcp.parcellate(m3_mtx, hcp.mmp)
mov4_p = hcp.parcellate(m4_mtx, hcp.mmp)

tsk1_p = hcp.parcellate(t1_mtx, hcp.mmp)
tsk2_p = hcp.parcellate(t2_mtx, hcp.mmp)
tsk3_p = hcp.parcellate(t3_mtx, hcp.mmp)
tsk4_p = hcp.parcellate(t4_mtx, hcp.mmp)
tsk5_p = hcp.parcellate(t5_mtx, hcp.mmp)
tsk6_p = hcp.parcellate(t6_mtx, hcp.mmp)

np.save('/data_qnap/yifeis/new/processed/'+str(subject)+'/rest1_p.npy', rst1_p)
np.save('/data_qnap/yifeis/new/processed/'+str(subject)+'/rest2_p.npy', rst2_p)
np.save('/data_qnap/yifeis/new/processed/'+str(subject)+'/rest3_p.npy', rst3_p)
np.save('/data_qnap/yifeis/new/processed/'+str(subject)+'/rest4_p.npy', rst4_p)

np.save('/data_qnap/yifeis/new/processed/'+str(subject)+'/movie1_p.npy', mov1_p)
np.save('/data_qnap/yifeis/new/processed/'+str(subject)+'/movie2_p.npy', mov2_p)
np.save('/data_qnap/yifeis/new/processed/'+str(subject)+'/movie3_p.npy', mov3_p)
np.save('/data_qnap/yifeis/new/processed/'+str(subject)+'/movie4_p.npy', mov4_p)

np.save('/data_qnap/yifeis/new/processed/'+str(subject)+'/retbar1_p.npy', tsk1_p)
np.save('/data_qnap/yifeis/new/processed/'+str(subject)+'/retbar2_p.npy', tsk2_p)
np.save('/data_qnap/yifeis/new/processed/'+str(subject)+'/retccw_p.npy', tsk3_p)
np.save('/data_qnap/yifeis/new/processed/'+str(subject)+'/retcon_p.npy', tsk4_p)
np.save('/data_qnap/yifeis/new/processed/'+str(subject)+'/retcw_p.npy', tsk5_p)
np.save('/data_qnap/yifeis/new/processed/'+str(subject)+'/retexp_p.npy', tsk6_p)
print("------Parcellation saved------")
## plotting
def save_plot_parcel():
    os.mkdir('/data_qnap/yifeis/ae_plots/'+str(subject))
    os.mkdir('/data_qnap/yifeis/ae_plots/'+str(subject)+'/movie')
    os.mkdir('/data_qnap/yifeis/ae_plots/'+str(subject)+'/movie/movie1')
    os.mkdir('/data_qnap/yifeis/ae_plots/'+str(subject)+'/movie/movie2')
    os.mkdir('/data_qnap/yifeis/ae_plots/'+str(subject)+'/movie/movie3')
    os.mkdir('/data_qnap/yifeis/ae_plots/'+str(subject)+'/movie/movie4')
    os.mkdir('/data_qnap/yifeis/ae_plots/'+str(subject)+'/rest')
    os.mkdir('/data_qnap/yifeis/ae_plots/'+str(subject)+'/rest/rest1')
    os.mkdir('/data_qnap/yifeis/ae_plots/'+str(subject)+'/rest/rest2')
    os.mkdir('/data_qnap/yifeis/ae_plots/'+str(subject)+'/rest/rest3')
    os.mkdir('/data_qnap/yifeis/ae_plots/'+str(subject)+'/rest/rest4')
    os.mkdir('/data_qnap/yifeis/ae_plots/'+str(subject)+'/task')
    os.mkdir('/data_qnap/yifeis/ae_plots/'+str(subject)+'/task/retbar1')
    os.mkdir('/data_qnap/yifeis/ae_plots/'+str(subject)+'/task/retbar2')
    os.mkdir('/data_qnap/yifeis/ae_plots/'+str(subject)+'/task/retccw')
    os.mkdir('/data_qnap/yifeis/ae_plots/'+str(subject)+'/task/retcon')
    os.mkdir('/data_qnap/yifeis/ae_plots/'+str(subject)+'/task/retcw')
    os.mkdir('/data_qnap/yifeis/ae_plots/'+str(subject)+'/task/retexp')
    print("------ Parcellated Surface Plots Saved in /data_qnap/yifeis/new_plots ------")
    nlp.view_surf(hcp.mesh.inflated,hcp.cortex_data(hcp.unparcellate(rst1_p[0], hcp.mmp)), bg_map=hcp.mesh.sulc).save_as_html('/data_qnap/yifeis/ae_plots/'+str(subject)+'/rest/rest1/parcellated.html')
    nlp.view_surf(hcp.mesh.inflated,hcp.cortex_data(hcp.unparcellate(rst2_p[0], hcp.mmp)), bg_map=hcp.mesh.sulc).save_as_html('/data_qnap/yifeis/ae_plots/'+str(subject)+'/rest/rest2/parcellated.html')
    nlp.view_surf(hcp.mesh.inflated,hcp.cortex_data(hcp.unparcellate(rst3_p[0], hcp.mmp)), bg_map=hcp.mesh.sulc).save_as_html('/data_qnap/yifeis/ae_plots/'+str(subject)+'/rest/rest3/parcellated.html')
    nlp.view_surf(hcp.mesh.inflated,hcp.cortex_data(hcp.unparcellate(rst4_p[0], hcp.mmp)), bg_map=hcp.mesh.sulc).save_as_html('/data_qnap/yifeis/ae_plots/'+str(subject)+'/rest/rest4/parcellated.html')

    nlp.view_surf(hcp.mesh.inflated,hcp.cortex_data(hcp.unparcellate(mov1_p[0], hcp.mmp)), bg_map=hcp.mesh.sulc).save_as_html('/data_qnap/yifeis/ae_plots/'+str(subject)+'/movie/movie1/parcellated.html')
    nlp.view_surf(hcp.mesh.inflated,hcp.cortex_data(hcp.unparcellate(mov2_p[0], hcp.mmp)), bg_map=hcp.mesh.sulc).save_as_html('/data_qnap/yifeis/ae_plots/'+str(subject)+'/movie/movie2/parcellated.html')
    nlp.view_surf(hcp.mesh.inflated,hcp.cortex_data(hcp.unparcellate(mov3_p[0], hcp.mmp)), bg_map=hcp.mesh.sulc).save_as_html('/data_qnap/yifeis/ae_plots/'+str(subject)+'/movie/movie3/parcellated.html')
    nlp.view_surf(hcp.mesh.inflated,hcp.cortex_data(hcp.unparcellate(mov4_p[0], hcp.mmp)), bg_map=hcp.mesh.sulc).save_as_html('/data_qnap/yifeis/ae_plots/'+str(subject)+'/movie/movie4/parcellated.html')

    nlp.view_surf(hcp.mesh.inflated,hcp.cortex_data(hcp.unparcellate(tsk1_p[0], hcp.mmp)), bg_map=hcp.mesh.sulc).save_as_html('/data_qnap/yifeis/ae_plots/'+str(subject)+'/task/retbar1/parcellated.html')
    nlp.view_surf(hcp.mesh.inflated,hcp.cortex_data(hcp.unparcellate(tsk2_p[0], hcp.mmp)), bg_map=hcp.mesh.sulc).save_as_html('/data_qnap/yifeis/ae_plots/'+str(subject)+'/task/retbar2/parcellated.html')
    nlp.view_surf(hcp.mesh.inflated,hcp.cortex_data(hcp.unparcellate(tsk3_p[0], hcp.mmp)), bg_map=hcp.mesh.sulc).save_as_html('/data_qnap/yifeis/ae_plots/'+str(subject)+'/task/retccw/parcellated.html')
    nlp.view_surf(hcp.mesh.inflated,hcp.cortex_data(hcp.unparcellate(tsk4_p[0], hcp.mmp)), bg_map=hcp.mesh.sulc).save_as_html('/data_qnap/yifeis/ae_plots/'+str(subject)+'/task/retcon/parcellated.html')
    nlp.view_surf(hcp.mesh.inflated,hcp.cortex_data(hcp.unparcellate(tsk5_p[0], hcp.mmp)), bg_map=hcp.mesh.sulc).save_as_html('/data_qnap/yifeis/ae_plots/'+str(subject)+'/task/retcw/parcellated.html')
    nlp.view_surf(hcp.mesh.inflated,hcp.cortex_data(hcp.unparcellate(tsk6_p[0], hcp.mmp)), bg_map=hcp.mesh.sulc).save_as_html('/data_qnap/yifeis/ae_plots/'+str(subject)+'/task/retexp/parcellated.html')
save_plot_parcel()

## ranking
def save_rank():
    os.mkdir('/data_qnap/yifeis/new/parcellation/'+str(subject))
    rst1_df = hcp.ranking(rst1_p[0], hcp.mmp)
    rst2_df = hcp.ranking(rst2_p[0], hcp.mmp)
    rst3_df = hcp.ranking(rst3_p[0], hcp.mmp)
    rst4_df = hcp.ranking(rst4_p[0], hcp.mmp)

    mov1_df = hcp.ranking(mov1_p[0], hcp.mmp)
    mov2_df = hcp.ranking(mov2_p[0], hcp.mmp)
    mov3_df = hcp.ranking(mov3_p[0], hcp.mmp)
    mov4_df = hcp.ranking(mov4_p[0], hcp.mmp)

    tsk1_df = hcp.ranking(tsk1_p[0], hcp.mmp)
    tsk2_df = hcp.ranking(tsk2_p[0], hcp.mmp)
    tsk3_df = hcp.ranking(tsk3_p[0], hcp.mmp)
    tsk4_df = hcp.ranking(tsk4_p[0], hcp.mmp)
    tsk5_df = hcp.ranking(tsk5_p[0], hcp.mmp)
    tsk6_df = hcp.ranking(tsk6_p[0], hcp.mmp)
    rst1_df.to_csv("/data_qnap/yifeis/new/parcellation/"+str(subject)+"/ranked_rest1_parcelated.csv")
    rst2_df.to_csv("/data_qnap/yifeis/new/parcellation/"+str(subject)+"/ranked_rest2_parcelated.csv")
    rst3_df.to_csv("/data_qnap/yifeis/new/parcellation/"+str(subject)+"/ranked_rest3_parcelated.csv")
    rst4_df.to_csv("/data_qnap/yifeis/new/parcellation/"+str(subject)+"/ranked_rest4_parcelated.csv")

    mov1_df.to_csv("/data_qnap/yifeis/new/parcellation/"+str(subject)+"/ranked_movie1_parcelated.csv")
    mov2_df.to_csv("/data_qnap/yifeis/new/parcellation/"+str(subject)+"/ranked_movie2_parcelated.csv")
    mov3_df.to_csv("/data_qnap/yifeis/new/parcellation/"+str(subject)+"/ranked_movie3_parcelated.csv")
    mov4_df.to_csv("/data_qnap/yifeis/new/parcellation/"+str(subject)+"/ranked_movie4_parcelated.csv")

    tsk1_df.to_csv("/data_qnap/yifeis/new/parcellation/"+str(subject)+"/ranked_retbar1_parcelated.csv")
    tsk2_df.to_csv("/data_qnap/yifeis/new/parcellation/"+str(subject)+"/ranked_retbar2_parcelated.csv")
    tsk3_df.to_csv("/data_qnap/yifeis/new/parcellation/"+str(subject)+"/ranked_retccw_parcelated.csv")
    tsk4_df.to_csv("/data_qnap/yifeis/new/parcellation/"+str(subject)+"/ranked_retcon_parcelated.csv")
    tsk5_df.to_csv("/data_qnap/yifeis/new/parcellation/"+str(subject)+"/ranked_retcw_parcelated.csv")
    tsk6_df.to_csv("/data_qnap/yifeis/new/parcellation/"+str(subject)+"/ranked_retexp_parcelated.csv")
save_rank()

# ------ Funcitonal Connectivity ------ #
correlation_measure = ConnectivityMeasure(kind='correlation')
r1_cor_mtx = correlation_measure.fit_transform([rst1_p])[0] # (379.379) numpy array
r2_cor_mtx = correlation_measure.fit_transform([rst2_p])[0]
r3_cor_mtx = correlation_measure.fit_transform([rst3_p])[0]
r4_cor_mtx = correlation_measure.fit_transform([rst4_p])[0]

m1_cor_mtx = correlation_measure.fit_transform([mov1_p])[0] # (379.379) numpy array
m2_cor_mtx = correlation_measure.fit_transform([mov2_p])[0]
m3_cor_mtx = correlation_measure.fit_transform([mov3_p])[0]
m4_cor_mtx = correlation_measure.fit_transform([mov4_p])[0]

t1_cor_mtx = correlation_measure.fit_transform([tsk1_p])[0] # (379.379) numpy array
t2_cor_mtx = correlation_measure.fit_transform([tsk2_p])[0]
t3_cor_mtx = correlation_measure.fit_transform([tsk3_p])[0]
t4_cor_mtx = correlation_measure.fit_transform([tsk4_p])[0]
t5_cor_mtx = correlation_measure.fit_transform([tsk5_p])[0]
t6_cor_mtx = correlation_measure.fit_transform([tsk6_p])[0]

nlp.plot_matrix(r1_cor_mtx, figure=(10, 8), labels=label_values[1:], reorder=False)
plt.title("Rest1 Correlation Matrix")
nlp.plot_matrix(m1_cor_mtx, figure=(10, 8), labels=label_values[1:], reorder=False)
plt.title("Movie1 Correlation Matrix")
nlp.plot_matrix(t1_cor_mtx, figure=(10, 8), labels=label_values[1:], reorder=False)
plt.title("Task1 Correlation Matrix")
# plt.show()
