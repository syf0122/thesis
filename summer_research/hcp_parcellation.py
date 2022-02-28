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

# get all subjects
dir = "/data_qnap/yifeis/new/"
subjects = os.listdir(dir)
subjects.remove("parcellation")
subjects.remove("processed")
subjects.sort()

session = ["REST1", "REST2", "REST3", "REST4"]
labels = hcp.mmp.labels
label_values = list(labels.values())
correlation_measure = ConnectivityMeasure(kind='correlation')
for subject in subjects[:20]:
    # Temproal Filtered
    r1 = nib.load(dir+"/processed/"+subject+"/temporal_filtered_rfMRI_REST1_7T_PA_Atlas_MSMAll_hp2000_clean.dtseries.nii")
    r2 = nib.load(dir+"/processed/"+subject+"/temporal_filtered_rfMRI_REST2_7T_AP_Atlas_MSMAll_hp2000_clean.dtseries.nii")
    r3 = nib.load(dir+"/processed/"+subject+"/temporal_filtered_rfMRI_REST3_7T_PA_Atlas_MSMAll_hp2000_clean.dtseries.nii")
    r4 = nib.load(dir+"/processed/"+subject+"/temporal_filtered_rfMRI_REST4_7T_AP_Atlas_MSMAll_hp2000_clean.dtseries.nii")
    r1 = r1.get_fdata()
    r2 = r2.get_fdata()
    r3 = r3.get_fdata()
    r4 = r4.get_fdata()

    rst1_p = hcp.parcellate(r1, hcp.mmp)
    rst2_p = hcp.parcellate(r2, hcp.mmp)
    rst3_p = hcp.parcellate(r3, hcp.mmp)
    rst4_p = hcp.parcellate(r4, hcp.mmp)

    r1_cor_mtx = correlation_measure.fit_transform([rst1_p])[0] # (379.379) numpy array
    r2_cor_mtx = correlation_measure.fit_transform([rst2_p])[0]
    r3_cor_mtx = correlation_measure.fit_transform([rst3_p])[0]
    r4_cor_mtx = correlation_measure.fit_transform([rst4_p])[0]
    print(r1_cor_mtx.shape)
    print(r2_cor_mtx.shape)
    print(r3_cor_mtx.shape)
    print(r4_cor_mtx.shape)
    nlp.plot_matrix(r1_cor_mtx, figure=(8, 6), reorder=False, cmap='jet')
    plt.title(subject + " " + session[0] + " Correlation Matrix (Temporal Filtered)")
    plt.savefig('/data_qnap/yifeis/NAS/parcellation/HCP/' + subject + "_" + session[0] + "_tf_Correlation_Matrix.png", bbox_inches='tight')
    nlp.plot_matrix(r2_cor_mtx, figure=(8, 6), reorder=False, cmap='jet')
    plt.title(subject + " " + session[1] + " Correlation Matrix (Temporal Filtered)")
    plt.savefig('/data_qnap/yifeis/NAS/parcellation/HCP/' + subject + "_" + session[1] + "_tf_Correlation_Matrix.png", bbox_inches='tight')
    nlp.plot_matrix(r3_cor_mtx, figure=(8, 6), reorder=False, cmap='jet')
    plt.title(subject + " " + session[2] + " Correlation Matrix (Temporal Filtered)")
    plt.savefig('/data_qnap/yifeis/NAS/parcellation/HCP/' + subject + "_" + session[2] + "_tf_Correlation_Matrix.png", bbox_inches='tight')
    nlp.plot_matrix(r4_cor_mtx, figure=(8, 6), reorder=False, cmap='jet')
    plt.title(subject + " " + session[3] + " Correlation Matrix (Temporal Filtered)")
    plt.savefig('/data_qnap/yifeis/NAS/parcellation/HCP/' + subject + "_" + session[3] + "_tf_Correlation_Matrix.png", bbox_inches='tight')

    # tf + norm
    r1 = hcp.normalize(r1)
    r2 = hcp.normalize(r2)
    r3 = hcp.normalize(r3)
    r4 = hcp.normalize(r4)

    rst1_p = hcp.parcellate(r1, hcp.mmp)
    rst2_p = hcp.parcellate(r2, hcp.mmp)
    rst3_p = hcp.parcellate(r3, hcp.mmp)
    rst4_p = hcp.parcellate(r4, hcp.mmp)

    r1_cor_mtx = correlation_measure.fit_transform([rst1_p])[0] # (379.379) numpy array
    r2_cor_mtx = correlation_measure.fit_transform([rst2_p])[0]
    r3_cor_mtx = correlation_measure.fit_transform([rst3_p])[0]
    r4_cor_mtx = correlation_measure.fit_transform([rst4_p])[0]
    print(r1_cor_mtx.shape)
    print(r2_cor_mtx.shape)
    print(r3_cor_mtx.shape)
    print(r4_cor_mtx.shape)
    nlp.plot_matrix(r1_cor_mtx, figure=(8, 6), reorder=False, cmap='jet')
    plt.title(subject + " " + session[0] + " Correlation Matrix (Temporal Filtered and Normalized)")
    plt.savefig('/data_qnap/yifeis/NAS/parcellation/HCP/' + subject + "_" + session[0] + "_norm_Correlation_Matrix.png", bbox_inches='tight')
    nlp.plot_matrix(r2_cor_mtx, figure=(8, 6), reorder=False, cmap='jet')
    plt.title(subject + " " + session[1] + " Correlation Matrix (Temporal Filtered and Normalized)")
    plt.savefig('/data_qnap/yifeis/NAS/parcellation/HCP/' + subject + "_" + session[1] + "_norm_Correlation_Matrix.png", bbox_inches='tight')
    nlp.plot_matrix(r3_cor_mtx, figure=(8, 6), reorder=False, cmap='jet')
    plt.title(subject + " " + session[2] + " Correlation Matrix (Temporal Filtered and Normalized)")
    plt.savefig('/data_qnap/yifeis/NAS/parcellation/HCP/' + subject + "_" + session[2] + "_norm_Correlation_Matrix.png", bbox_inches='tight')
    nlp.plot_matrix(r4_cor_mtx, figure=(8, 6), reorder=False, cmap='jet')
    plt.title(subject + " " + session[3] + " Correlation Matrix (Temporal Filtered and Normalized)")
    plt.savefig('/data_qnap/yifeis/NAS/parcellation/HCP/' + subject + "_" + session[3] + "_norm_Correlation_Matrix.png", bbox_inches='tight')

    # raw
    r1 = nib.load(dir+subject+"/MNINonLinear/Results/rfMRI_REST1_7T_PA/rfMRI_REST1_7T_PA_Atlas_MSMAll_hp2000_clean.dtseries.nii")
    r2 = nib.load(dir+subject+"/MNINonLinear/Results/rfMRI_REST2_7T_AP/rfMRI_REST2_7T_AP_Atlas_MSMAll_hp2000_clean.dtseries.nii")
    r3 = nib.load(dir+subject+"/MNINonLinear/Results/rfMRI_REST3_7T_PA/rfMRI_REST3_7T_PA_Atlas_MSMAll_hp2000_clean.dtseries.nii")
    r4 = nib.load(dir+subject+"/MNINonLinear/Results/rfMRI_REST4_7T_AP/rfMRI_REST4_7T_AP_Atlas_MSMAll_hp2000_clean.dtseries.nii")
    r1 = r1.get_fdata()
    r2 = r2.get_fdata()
    r3 = r3.get_fdata()
    r4 = r4.get_fdata()

    rst1_p = hcp.parcellate(r1, hcp.mmp)
    rst2_p = hcp.parcellate(r2, hcp.mmp)
    rst3_p = hcp.parcellate(r3, hcp.mmp)
    rst4_p = hcp.parcellate(r4, hcp.mmp)

    r1_cor_mtx = correlation_measure.fit_transform([rst1_p])[0] # (379.379) numpy array
    r2_cor_mtx = correlation_measure.fit_transform([rst2_p])[0]
    r3_cor_mtx = correlation_measure.fit_transform([rst3_p])[0]
    r4_cor_mtx = correlation_measure.fit_transform([rst4_p])[0]
    print(r1_cor_mtx.shape)
    print(r2_cor_mtx.shape)
    print(r3_cor_mtx.shape)
    print(r4_cor_mtx.shape)
    nlp.plot_matrix(r1_cor_mtx, figure=(8, 6), reorder=False, cmap='jet')
    plt.title(subject + " " + session[0] + " Correlation Matrix")
    plt.savefig('/data_qnap/yifeis/NAS/parcellation/HCP/' + subject + "_" + session[0] + "_Correlation_Matrix.png", bbox_inches='tight')
    nlp.plot_matrix(r2_cor_mtx, figure=(8, 6), reorder=False, cmap='jet')
    plt.title(subject + " " + session[1] + " Correlation Matrix")
    plt.savefig('/data_qnap/yifeis/NAS/parcellation/HCP/' + subject + "_" + session[1] + "_Correlation_Matrix.png", bbox_inches='tight')
    nlp.plot_matrix(r3_cor_mtx, figure=(8, 6), reorder=False, cmap='jet')
    plt.title(subject + " " + session[2] + " Correlation Matrix")
    plt.savefig('/data_qnap/yifeis/NAS/parcellation/HCP/' + subject + "_" + session[2] + "_Correlation_Matrix.png", bbox_inches='tight')
    nlp.plot_matrix(r4_cor_mtx, figure=(8, 6), reorder=False, cmap='jet')
    plt.title(subject + " " + session[3] + " Correlation Matrix")
    plt.savefig('/data_qnap/yifeis/NAS/parcellation/HCP/' + subject + "_" + session[3] + "_Correlation_Matrix.png", bbox_inches='tight')
