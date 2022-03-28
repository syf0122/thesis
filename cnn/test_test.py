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

# def mse(input, output):
#     difference_array = np.subtract(input, output)
#     squared_array = np.square(difference_array)
#     mse = squared_array.mean(axis=1)
#     return mse
#
# def inference(ts, model):
#     feats = ts.to(device)
#     with torch.no_grad():
#         prediction = model(feats)
#     pred = prediction.cpu().numpy()
#     pred = pred.squeeze()
#     return pred
#
#
# model = Unet_160k(1, 1)
# device = torch.device('cuda:0')
# model_path = "/data_qnap/yifeis/spherical_cnn/test/Unet_160k_test_final.pkl"
# model.to(device)
# model.load_state_dict(torch.load(model_path))
# model.eval()
#
#
# # load the train data
# train_data_dir = '/data_qnap/yifeis/spherical_cnn/test/first_10_train_data.npy'
# t_series = np.load(train_data_dir)
# data = torch.from_numpy(t_series).unsqueeze(1) # 163842, 1, 36000
# pred_t_series = np.zeros(t_series.shape) # 163842, 36000
#
# ## prediction
# progress_bar = tqdm(range((data.size()[2])))
# for i in range(data.size()[2]):
#     target = data[:, :, i]
#     pred = inference(target, model) # 163438,
#     pred_t_series[:, i] = pred
#     progress_bar.update(1)
# progress_bar.close()
#
# # calculate the MSELoss
# loss = mse(t_series, pred_t_series)
# print(loss.shape)
# np.save('/data_qnap/yifeis/spherical_cnn/test/mse_train_data.npy', loss)

results = np.load('/data_qnap/yifeis/spherical_cnn/test/mse_train_data.npy')
results_df = pd.DataFrame(results)
results_df.to_csv("/data_qnap/yifeis/spherical_cnn/test/mse_train_data.csv")
