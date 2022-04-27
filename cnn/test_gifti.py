import torch
import torch.nn as nn
import argparse
import torchvision
import numpy as np
import glob
import os
from tqdm import tqdm
import pandas as pd
import nibabel as nib
import hcp_utils as hcp

from model import Unet_160k
from sphericalunet.utils.vtk import read_vtk, write_vtk, resample_label
from sphericalunet.utils.interp_numpy import resampleSphereSurf

def se(input, output):
    difference_array = np.subtract(input, output)
    squared_array = np.square(difference_array)
    return squared_array

def mse(input, output):
    difference_array = np.subtract(input, output)
    squared_array = np.square(difference_array)
    mse = squared_array.mean(axis=1)
    return mse

def inference(ts, model):
    feats = ts.to(device)
    with torch.no_grad():
        prediction = model(feats)
    pred = prediction.cpu().numpy()
    pred = pred.squeeze()
    return pred

#
# f_img = nib.load('/data_qnap/yifeis/NAS/HCP_7T/100610/movie1_left.func.gii')
# f_data = [x.data for x in f_img.darrays]
# f_data = np.reshape(f_data,(len(f_data[0]),len(f_data)))
# v_data = read_vtk('/data_qnap/yifeis/NAS/HCP_7T/100610/movie1_left.vtk')['cdata']
#
# print(f_data.shape)
# print(v_data.shape)
# print(f_data.max())
# print(v_data.max())
# print(f_data.min())
# print(v_data.min())
# print(f_data.mean())
# print(v_data.mean())
# quit()

sub_train = 20
hemi = 'right'

model = Unet_160k(1, 1)
device = torch.device('cuda:0')
model_path = "/data_qnap/yifeis/spherical_cnn/models/epo_20_20/Unet_160k_test_gifti_"+str(sub_train)+'_'+hemi+"_final.pkl"
print(f"Use model at :{model_path} ")
model.to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# get all subjects
subjects = os.listdir('/data_qnap/yifeis/NAS/HCP_7T') # 1 - 50
# subjects = os.listdir('/data_qnap/yifeis/HCP_7T') # 51-100
subjects.sort()
subjects = subjects[20:]
sessions = ['movie1', 'movie2', 'movie3', 'movie4',
            'rest1', 'rest2', 'rest3', 'rest4',
            'retbar1', 'retbar2', 'retccw', 'retcon', 'retcw', 'retexp']
print(subjects)
for sub in subjects:
    for ses in sessions:
        # load the test data
        test_data_dir = '/data_qnap/yifeis/NAS/HCP_7T/'+sub+'/'+ses+'_'+hemi+'.func.gii'
        # test_data_dir = '/data_qnap/yifeis/HCP_7T/'+sub+'/'+ses+'_'+hemi+'.func.gii'
        print(test_data_dir)
        # read gifti
        f_data = nib.load(test_data_dir)
        f_data = np.array(f_data.agg_data()) # tp, 164k
        # temporal normalization
        f_data = hcp.normalize(f_data)
        f_data = f_data.T # 164k, tp
        f_data = np.nan_to_num(f_data) # nan to 0
        data = torch.from_numpy(f_data).unsqueeze(1) # 163842, 1, # of timepoints
        pred_t_series = np.zeros(f_data.shape) # 163842, # of timepoints

        ## prediction
        progress_bar = tqdm(range((data.size()[2])))
        for i in range(data.size()[2]):
            target = data[:, :, i]
            pred = inference(target, model) # 163438,
            pred_t_series[:, i] = pred
            progress_bar.update(1)
        progress_bar.close()

        # calculate the MSE and se
        se_loss = se(f_data, pred_t_series)
        mse_loss = mse(f_data, pred_t_series)
        print(se_loss.shape)
        print(mse_loss.shape)
        np.save('/data_qnap/yifeis/spherical_cnn/results/MSE/'+str(sub)+'_'+ses+'_'+hemi+'_using_gifti.npy', mse_loss)
        np.save('/data_qnap/yifeis/spherical_cnn/results/SE/'+str(sub)+'_'+ses+'_'+hemi+'_using_gifti.npy', se_loss)
