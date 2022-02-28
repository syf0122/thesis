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

subjects = [100610, 102311, 102816, 104416, 105923,
            108323, 109123, 111514, 114823, 115017,
            115825, 116726, 118225, 125525, 126426,
            128935, 130114, 130518, 131217, 131722,
            132118, 134627, 134829, 135124, 137128,
            140117, 144226, 145834, 146129, 146432,
            146735, 146937, 148133, 150423, 155938,
            156334, 157336, 158035, 158136, 159239,
            162935, 164131, 164636, 165436, 167036,
            167440, 169040, 169343, 169444, 169747]
for subject in subjects:
    mtx1_p = np.load('/data_qnap/yifeis/new/processed/'+str(subject)+'/movie1_p.npy')
    mtx2_p = np.load('/data_qnap/yifeis/new/processed/'+str(subject)+'/movie2_p.npy')
    mtx3_p = np.load('/data_qnap/yifeis/new/processed/'+str(subject)+'/movie3_p.npy')
    mtx4_p = np.load('/data_qnap/yifeis/new/processed/'+str(subject)+'/movie4_p.npy')

    mtx5_p = np.load('/data_qnap/yifeis/new/processed/'+str(subject)+'/rest1_p.npy')
    mtx6_p = np.load('/data_qnap/yifeis/new/processed/'+str(subject)+'/rest2_p.npy')
    mtx7_p = np.load('/data_qnap/yifeis/new/processed/'+str(subject)+'/rest3_p.npy')
    mtx8_p = np.load('/data_qnap/yifeis/new/processed/'+str(subject)+'/rest4_p.npy')

    mtx9_p = np.load('/data_qnap/yifeis/new/processed/'+str(subject)+'/retbar1_p.npy')
    mtx10_p = np.load('/data_qnap/yifeis/new/processed/'+str(subject)+'/retbar2_p.npy')
    mtx11_p = np.load('/data_qnap/yifeis/new/processed/'+str(subject)+'/retccw_p.npy')
    mtx12_p = np.load('/data_qnap/yifeis/new/processed/'+str(subject)+'/retcon_p.npy')
    mtx13_p = np.load('/data_qnap/yifeis/new/processed/'+str(subject)+'/retcw_p.npy')
    mtx14_p = np.load('/data_qnap/yifeis/new/processed/'+str(subject)+'/retexp_p.npy')

    mtx1_p = (mtx1_p - mtx1_p.min()) / (mtx1_p.max() - mtx1_p.min())
    mtx2_p = (mtx2_p - mtx2_p.min()) / (mtx2_p.max() - mtx2_p.min())
    mtx3_p = (mtx3_p - mtx3_p.min()) / (mtx3_p.max() - mtx3_p.min())
    mtx4_p = (mtx4_p - mtx4_p.min()) / (mtx4_p.max() - mtx4_p.min())

    mtx5_p = (mtx5_p - mtx5_p.min()) / (mtx5_p.max() - mtx5_p.min())
    mtx6_p = (mtx6_p - mtx6_p.min()) / (mtx6_p.max() - mtx6_p.min())
    mtx7_p = (mtx7_p - mtx7_p.min()) / (mtx7_p.max() - mtx7_p.min())
    mtx8_p = (mtx8_p - mtx8_p.min()) / (mtx8_p.max() - mtx8_p.min())

    mtx9_p = (mtx9_p - mtx9_p.min()) / (mtx9_p.max() - mtx9_p.min())
    mtx10_p = (mtx10_p - mtx10_p.min()) / (mtx10_p.max() - mtx10_p.min())
    mtx11_p = (mtx11_p - mtx11_p.min()) / (mtx11_p.max() - mtx11_p.min())
    mtx12_p = (mtx12_p - mtx12_p.min()) / (mtx12_p.max() - mtx12_p.min())
    mtx13_p = (mtx13_p - mtx13_p.min()) / (mtx13_p.max() - mtx13_p.min())
    mtx14_p = (mtx14_p - mtx14_p.min()) / (mtx14_p.max() - mtx14_p.min())

    nlp.view_surf(hcp.mesh.inflated,hcp.cortex_data(hcp.unparcellate(mtx1_p[0], hcp.mmp)), bg_map=hcp.mesh.sulc, symmetric_cmap = False, vmax = 1, vmin = 0).save_as_html('/data_qnap/yifeis/ae_plots/'+str(subject)+'/movie/movie1/parcellated_norm.html')
    nlp.view_surf(hcp.mesh.inflated,hcp.cortex_data(hcp.unparcellate(mtx2_p[0], hcp.mmp)), bg_map=hcp.mesh.sulc, symmetric_cmap = False, vmax = 1, vmin = 0).save_as_html('/data_qnap/yifeis/ae_plots/'+str(subject)+'/movie/movie2/parcellated_norm.html')
    nlp.view_surf(hcp.mesh.inflated,hcp.cortex_data(hcp.unparcellate(mtx3_p[0], hcp.mmp)), bg_map=hcp.mesh.sulc, symmetric_cmap = False, vmax = 1, vmin = 0).save_as_html('/data_qnap/yifeis/ae_plots/'+str(subject)+'/movie/movie3/parcellated_norm.html')
    nlp.view_surf(hcp.mesh.inflated,hcp.cortex_data(hcp.unparcellate(mtx4_p[0], hcp.mmp)), bg_map=hcp.mesh.sulc, symmetric_cmap = False, vmax = 1, vmin = 0).save_as_html('/data_qnap/yifeis/ae_plots/'+str(subject)+'/movie/movie4/parcellated_norm.html')

    nlp.view_surf(hcp.mesh.inflated,hcp.cortex_data(hcp.unparcellate(mtx5_p[0], hcp.mmp)), bg_map=hcp.mesh.sulc, symmetric_cmap = False, vmax = 1, vmin = 0).save_as_html('/data_qnap/yifeis/ae_plots/'+str(subject)+'/rest/rest1/parcellated_norm.html')
    nlp.view_surf(hcp.mesh.inflated,hcp.cortex_data(hcp.unparcellate(mtx6_p[0], hcp.mmp)), bg_map=hcp.mesh.sulc, symmetric_cmap = False, vmax = 1, vmin = 0).save_as_html('/data_qnap/yifeis/ae_plots/'+str(subject)+'/rest/rest2/parcellated_norm.html')
    nlp.view_surf(hcp.mesh.inflated,hcp.cortex_data(hcp.unparcellate(mtx7_p[0], hcp.mmp)), bg_map=hcp.mesh.sulc, symmetric_cmap = False, vmax = 1, vmin = 0).save_as_html('/data_qnap/yifeis/ae_plots/'+str(subject)+'/rest/rest3/parcellated_norm.html')
    nlp.view_surf(hcp.mesh.inflated,hcp.cortex_data(hcp.unparcellate(mtx8_p[0], hcp.mmp)), bg_map=hcp.mesh.sulc, symmetric_cmap = False, vmax = 1, vmin = 0).save_as_html('/data_qnap/yifeis/ae_plots/'+str(subject)+'/rest/rest4/parcellated_norm.html')

    nlp.view_surf(hcp.mesh.inflated,hcp.cortex_data(hcp.unparcellate(mtx9_p[0], hcp.mmp)), bg_map=hcp.mesh.sulc, symmetric_cmap = False, vmax = 1, vmin = 0).save_as_html('/data_qnap/yifeis/ae_plots/'+str(subject)+'/task/retbar1/parcellated_norm.html')
    nlp.view_surf(hcp.mesh.inflated,hcp.cortex_data(hcp.unparcellate(mtx10_p[0], hcp.mmp)), bg_map=hcp.mesh.sulc, symmetric_cmap = False, vmax = 1, vmin = 0).save_as_html('/data_qnap/yifeis/ae_plots/'+str(subject)+'/task/retbar2/parcellated_norm.html')
    nlp.view_surf(hcp.mesh.inflated,hcp.cortex_data(hcp.unparcellate(mtx11_p[0], hcp.mmp)), bg_map=hcp.mesh.sulc, symmetric_cmap = False, vmax = 1, vmin = 0).save_as_html('/data_qnap/yifeis/ae_plots/'+str(subject)+'/task/retccw/parcellated_norm.html')
    nlp.view_surf(hcp.mesh.inflated,hcp.cortex_data(hcp.unparcellate(mtx12_p[0], hcp.mmp)), bg_map=hcp.mesh.sulc, symmetric_cmap = False, vmax = 1, vmin = 0).save_as_html('/data_qnap/yifeis/ae_plots/'+str(subject)+'/task/retcon/parcellated_norm.html')
    nlp.view_surf(hcp.mesh.inflated,hcp.cortex_data(hcp.unparcellate(mtx13_p[0], hcp.mmp)), bg_map=hcp.mesh.sulc, symmetric_cmap = False, vmax = 1, vmin = 0).save_as_html('/data_qnap/yifeis/ae_plots/'+str(subject)+'/task/retcw/parcellated_norm.html')
    nlp.view_surf(hcp.mesh.inflated,hcp.cortex_data(hcp.unparcellate(mtx14_p[0], hcp.mmp)), bg_map=hcp.mesh.sulc, symmetric_cmap = False, vmax = 1, vmin = 0).save_as_html('/data_qnap/yifeis/ae_plots/'+str(subject)+'/task/retexp/parcellated_norm.html')
