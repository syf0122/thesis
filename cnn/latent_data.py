import torch
import torch.nn as nn
import argparse
import torchvision
import numpy as np
import glob
import os
from tqdm import tqdm
import pandas as pd
from model import Unet_160k
import nibabel as nib
import hcp_utils as hcp
import nilearn.plotting as plotting
from sphericalunet.utils.vtk import read_vtk, write_vtk, resample_label

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook
hemi = 'left'
model = Unet_160k(1, 1)
device = torch.device('cpu')
model_path = "/data_qnap/yifeis/spherical_cnn/models/epo_20_20/Unet_160k_test_gifti_20_"+hemi+"_final.pkl"
print(f"Use model at :{model_path} ")
model.to(device)
model.load_state_dict(torch.load(model_path, map_location=device))

# get all subjects
# subjects = os.listdir('/data_qnap/yifeis/NAS/HCP_7T') # 1 - 50
subjects = os.listdir('/data_qnap/yifeis/HCP_7T') # 51-100
subjects.sort()
sessions = ['movie1', 'movie2', 'movie3', 'movie4',
            'rest1',  'rest2',  'rest3',  'rest4',
            'retbar1', 'retbar2', 'retccw', 'retcw', 'retcon', 'retexp']

# print(subjects)
# for sub in subjects:
#     for ses in sessions:
#         # load the test data
#         # test_data_dir = '/data_qnap/yifeis/NAS/HCP_7T/'+sub+'/'+ses+'_'+hemi+'.func.gii'
#         test_data_dir = '/data_qnap/yifeis/HCP_7T/'+sub+'/'+ses+'_'+hemi+'.func.gii'
#         print(test_data_dir)
#         # read gifti
#         f_data = nib.load(test_data_dir)
#         # Visualize feature maps
#         activation = {}
#
#         f_data = np.array(f_data.agg_data()) # tp, 164k
#         # temporal normalization
#         f_data = hcp.normalize(f_data)
#         f_data = f_data.T # 164k, tp
#         f_data = np.nan_to_num(f_data) # nan to 0
#         data = torch.from_numpy(f_data).unsqueeze(1) # 163842, 1, # of timepoints
#
#         model.down5.register_forward_hook(get_activation('down5'))
#
#         data = data[:, :, 0] # at first timepoint
#         output = model(data)
#         act = activation['down5'].squeeze()
#         latent = act.numpy().mean(axis=1)
#         print(latent.shape)
#         # surface data
#         surf_data = read_vtk('/data_qnap/yifeis/spherical_cnn/code/sphere/sphere_642_rotated_0.vtk')
#         surf_data['sCNN_latent_data'] = latent
#         write_vtk(surf_data,'/data_qnap/yifeis/spherical_cnn/results/latent/'+hemi+'_'+sub+'_'+ses+'_642.vtk')


# baseline
activation = {}
data = np.ones((163842, 900)).astype(np.float64)
data = torch.from_numpy(data).unsqueeze(1) # 163842, 1, 900
model.down5.register_forward_hook(get_activation('down5'))
data = data[:, :, 0] # at the first timepoint
output = model(data.float())
act = activation['down5'].squeeze()
latent = act.numpy().mean(axis=1)
print(latent.shape)
# surface data
surf_data = read_vtk('/data_qnap/yifeis/spherical_cnn/code/sphere/sphere_642_rotated_0.vtk')
surf_data['sCNN_latent_data'] = latent
write_vtk(surf_data,'/data_qnap/yifeis/spherical_cnn/results/latent/baseline_642.vtk')
