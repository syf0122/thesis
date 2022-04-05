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

sub_train = 20
hemi = 'right'

model = Unet_160k(1, 1)
device = torch.device('cuda:0')
model_path = "/data_qnap/yifeis/spherical_cnn/test/epo_5/Unet_160k_test_"+str(sub_train)+'_'+hemi+"_final.pkl"
print(f"Use model at :{model_path} ")
model.to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# get all subjects
subjects = os.listdir('/data_qnap/yifeis/NAS/HCP_7T')
subjects.sort()
sessions = ['movie1', 'movie2', 'movie3', 'movie4']

for sub in subjects[:30]:
    for ses in sessions:
        # load the test data
        test_data_dir = '/data_qnap/yifeis/NAS/HCP_7T/'+sub+'/'+ses+'_'+hemi+'.vtk'
        print(test_data_dir)
        # read vtk
        f_data = read_vtk(test_data_dir)['cdata']
        # normalization
        f_data = (f_data - np.mean(f_data,axis=0))/np.std(f_data,axis=0)
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
        np.save('/data_qnap/yifeis/spherical_cnn/results/MSE/'+str(sub)+'_'+ses+'_'+hemi+'.npy', mse_loss)
        np.save('/data_qnap/yifeis/spherical_cnn/results/SE/'+str(sub)+'_'+ses+'_'+hemi+'.npy', se_loss)

        # mse_df = pd.DataFrame(mse_loss)
        # mse_df.to_csv("/data_qnap/yifeis/spherical_cnn/results/MSE/"+str(sub)+'_'+ses+'_'+hemi+".csv")
        # se_df = pd.DataFrame(se_loss)
        # se_df.to_csv("/data_qnap/yifeis/spherical_cnn/results/SE/"+str(sub)+'_'+ses+'_'+hemi+".csv")
