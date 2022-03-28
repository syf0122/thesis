#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 22:46:50 2019

@author: Fenqiang Zhao, https://github.com/zhaofenqiang

Contact: zhaofenqiang0221@gmail.com
"""

import torch
import argparse
import torchvision
import numpy as np
import glob
import os

from model import Unet_40k, Unet_160k
from sphericalunet.utils.vtk import read_vtk, write_vtk, resample_label
from sphericalunet.utils.utils import get_par_36_to_fs_vec
from sphericalunet.utils.interp_numpy import resampleSphereSurf

class BrainSphere(torch.utils.data.Dataset):

    def __init__(self, root1):

        self.files = sorted(glob.glob(os.path.join(root1, '*.vtk')))

    def __getitem__(self, index):
        file = self.files[index]
        data = read_vtk(file)

        return data, file

    def __len__(self):
        return len(self.files)


def inference(ts, model):
    feats = ts.to(device)
    with torch.no_grad():
        prediction = model(feats)
    pred = prediction.cpu().numpy()
    return pred


in_file = ''
hemi = 'left'
device = torch.device('cuda:0')
out_file = in_file[0:-4] + '.recon.vtk'
model = Unet_160k(1, 1)
model_path = '160k_model.pkl'
n_vertices = 163842

model_path = 'trained_models/' + hemi + '_hemi_' +  model_path
model.to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

par_36_to_fs_vec = get_par_36_to_fs_vec()

template = read_vtk('neigh_indices/sphere_' + str(n_vertices) + '.vtk')

if in_file is not None:
    orig_data = read_vtk(in_file)
    tseries_data = orig_data['cmap'][0:n_vertices]
    tseries_data = torch.from_numpy(tseries_data).unsqueeze(1)

    pred = inference(tseries_data, model)
    pred = par_36_to_fs_vec[pred]

    orig_lbl = resample_label(template['vertices'], orig_data['vertices'], pred)

    orig_data['par_fs_vec'] = orig_lbl
    write_vtk(orig_data, out_file)
