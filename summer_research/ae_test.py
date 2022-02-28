import os
import sys
import pathlib
import shutil
import tempfile
import random
import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import nibabel as nib
import hcp_utils as hcp
from nilearn import plotting as nlp
from nilearn import surface as nsf
from nilearn.connectome import ConnectivityMeasure
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model

'''
    Load data
'''
def load_OAS_data(gp, subjects, resolution, prepro):
    dir = '/data_qnap/yifeis/NAS/data/'+gp+'_p/'
    processed_files = os.listdir(dir)
    processed_files.sort()
    sessions = {}
    data = {}
    for subject in subjects:
        sub_data = []
        sub_session = []
        for f in processed_files:
            if subject in f and ("_"+resolution+"_"+prepro+"_p.npy") in f:
                mtx_p = np.load(dir + f) # (149, 360)
                mtx_p = (mtx_p - mtx_p.min()) / (mtx_p.max() - mtx_p.min()) # normalize to 0-1
                sub_data.append(mtx_p)
                sub_session.append(f[13:18])
        data[subject] = sub_data
        sessions[subject] = sub_session
    return (data, sessions)

def load_HCP_data(subjects):
    dir = '/data_qnap/yifeis/new/processed/'
    sessions = {}
    data = {}
    data_360 = {}
    for subject in subjects:
        sub_data = []
        sub_data_360 = []
        sub_session = []
        processed_files = os.listdir(dir + subject + "/")
        processed_files.sort()
        for f in processed_files:
            if "rest" in f and "_p.npy" in f:
                mtx_p = np.load(dir + subject + "/" + f) # (900, 379)
                mtx_p = (mtx_p - mtx_p.min()) / (mtx_p.max() - mtx_p.min()) # normalize to 0-1
                sub_data.append(mtx_p)
                sub_data_360.append(mtx_p[:, :360])
                sub_session.append(f[:5])
        data[subject] = sub_data
        data_360[subject] = sub_data_360
        sessions[subject] = sub_session
    return (data, data_360, sessions)

'''
    Test using the loaded models
    Get reconstructed data and also the latent data
'''
def test(encoder, decoder, test_data):
    # latent_vector
    latent_vector = encoder.predict(test_data)
    # decoder output
    reconstructed_data = decoder.predict(latent_vector)
    return (latent_vector, reconstructed_data)

'''
    Calculate the MSE
    Regional MSE
'''
def mse(input, output):
    difference_array = np.subtract(input, output)
    squared_array = np.square(difference_array)
    mse = squared_array.mean()
    return mse

def mse_by_region(input, recon):
    mse_all = {}
    for sub in input: # for each subject
        subject_i = input[sub]
        subject_o = recon[sub]
        mse_sub = []
        for n in range(len(subject_i)):# for each session
            mtx_i = subject_i[n]
            mtx_o = subject_o[n]
            mse_ses = np.zeros(mtx_i.shape[1])
            for i in range(mtx_i.shape[1]): # for 360/379 region
                region_i =  mtx_i[:, i]
                region_o =  mtx_o[:, i]
                mse_ses[i] = mse(region_i, region_o)
            mse_sub.append(mse_ses)
        mse_all[sub] = mse_sub
    return mse_all

'''
    Load Trained Models
'''
autoencoder_360 = load_model('/data_qnap/yifeis/NAS/ae/train_hc/autoencoder_360.hdf5', compile=False)
encoder_360 = load_model('/data_qnap/yifeis/NAS/ae/train_hc/encoder_360.hdf5', compile=False)
decoder_360 = load_model('/data_qnap/yifeis/NAS/ae/train_hc/decoder_360.hdf5', compile=False)

autoencoder_379 = load_model('/data_qnap/yifeis/NAS/ae/train_hc/autoencoder_379.hdf5', compile=False)
encoder_379 = load_model('/data_qnap/yifeis/NAS/ae/train_hc/encoder_379.hdf5', compile=False)
decoder_379 = load_model('/data_qnap/yifeis/NAS/ae/train_hc/decoder_379.hdf5', compile=False)
print("Loading Successful!")
print(autoencoder_360.summary())
print(autoencoder_379.summary())


if len(sys.argv) == 3:
    print("Test on OAS data")
    '''
        Load Testing data
    '''
    group  = sys.argv[1].upper()
    prepro = sys.argv[2].lower()
    if group not in ["AD", "HC"]:
        group = "HC"
    if prepro not in ['norm', 'tf']:
        prepro = 'norm'

    # get all subjects
    test_subjects = os.listdir("/data_qnap/yifeis/NAS/data/"+group+"/")
    test_subjects.sort()

    # load data
    test_data_360_2mm, test_sessions_360_2mm = load_OAS_data(group, test_subjects[:50], "2mm", prepro)
    test_data_360_4mm, test_sessions_360_4mm = load_OAS_data(group, test_subjects[:50], "4mm", prepro)
    test_data_379, test_sessions_379 = load_OAS_data(group, test_subjects[:50], 'surf', prepro)

    '''
        Test
    '''
    test_latent_data_360_2mm = {}
    test_recon_data_360_2mm  = {}

    test_latent_data_360_4mm = {}
    test_recon_data_360_4mm  = {}

    test_latent_data_379 = {}
    test_recon_data_379  = {}

    for sub in test_subjects[:50]:
        # get the test data for each subject
        sub_360_2mm = test_data_360_2mm[sub]
        sub_360_4mm = test_data_360_4mm[sub]
        sub_379     = test_data_379[sub]

        sub_l_2 = []
        sub_r_2 = []

        sub_l_4 = []
        sub_r_4 = []

        sub_l_s = []
        sub_r_s = []

        for mtx in sub_360_2mm: # (149, 360)
            latent, recon = test(encoder_360, decoder_360, mtx)
            sub_l_2.append(latent)
            sub_r_2.append(recon)

        for mtx in sub_360_4mm: # (149, 360)
            latent, recon = test(encoder_360, decoder_360, mtx)
            sub_l_4.append(latent)
            sub_r_4.append(recon)

        for mtx in sub_379: # (149, 379)
            latent, recon = test(encoder_379, decoder_379, mtx)
            sub_l_s.append(latent)
            sub_r_s.append(recon)

        test_latent_data_360_2mm[sub] = sub_l_2
        test_recon_data_360_2mm[sub] = sub_r_2

        test_latent_data_360_4mm[sub] = sub_l_4
        test_recon_data_360_4mm[sub] = sub_r_4

        test_latent_data_379[sub] = sub_l_s
        test_recon_data_379[sub] = sub_r_s

    print('test finished!')

    '''
        Save reconstruction results
        Save latent variables
    '''

    for sub in test_subjects[:50]:
        # volume 2mm
        test_re = test_recon_data_360_2mm[sub]           # reconstructed data for sub
        test_latent_re = test_latent_data_360_2mm[sub]   # latent data for sub
        test_ses_re = test_sessions_360_2mm[sub]         # session information for sub
        # save reconstructed/latent data
        for i in range(len(test_re)):
            run = test_ses_re[i]
            # save reconstructed data
            mtx_recon_df = pd.DataFrame(data=test_re[i]) # (149, 360)
            mtx_recon_df.to_csv("/data_qnap/yifeis/NAS/ae/results/model_hc_results/recon/"+group+"_"+str(sub)+"_"+run+"_2mm_"+prepro+"_recon_df.csv")
            # save latent data
            mtx_latent_df = pd.DataFrame(data=test_latent_re[i]) # (149, 45)
            mtx_latent_df.to_csv("/data_qnap/yifeis/NAS/ae/results/model_hc_results/latent/"+group+"_"+str(sub)+"_"+run+"_2mm_"+prepro+"_latent_df.csv")

        # volume 4mm
        test_re = test_recon_data_360_4mm[sub]           # reconstructed data for sub
        test_latent_re = test_latent_data_360_4mm[sub]   # latent data for sub
        test_ses_re = test_sessions_360_4mm[sub]         # session information for sub
        # save reconstructed/latent data
        for i in range(len(test_re)):
            run = test_ses_re[i]
            # save reconstructed data
            mtx_recon_df = pd.DataFrame(data=test_re[i]) # (149, 360)
            mtx_recon_df.to_csv("/data_qnap/yifeis/NAS/ae/results/model_hc_results/recon/"+group+"_"+str(sub)+"_"+run+"_4mm_"+prepro+"_recon_df.csv")
            # save latent data
            mtx_latent_df = pd.DataFrame(data=test_latent_re[i]) # (149, 45)
            mtx_latent_df.to_csv("/data_qnap/yifeis/NAS/ae/results/model_hc_results/latent/"+group+"_"+str(sub)+"_"+run+"_4mm_"+prepro+"_latent_df.csv")

        # surface
        test_re = test_recon_data_379[sub]           # reconstructed data for sub
        test_latent_re = test_latent_data_379[sub]   # latent data for sub
        test_ses_re = test_sessions_379[sub]         # session information for sub
        # save reconstructed/latent data
        for i in range(len(test_re)):
            run = test_ses_re[i]
            # save reconstructed data
            mtx_recon_df = pd.DataFrame(data=test_re[i])  # (149, 379)
            mtx_recon_df.to_csv("/data_qnap/yifeis/NAS/ae/results/model_hc_results/recon/"+group+"_"+str(sub)+"_"+run+"_surf_"+prepro+"_recon_df.csv")
            # save latent data
            mtx_latent_df = pd.DataFrame(data=test_latent_re[i]) # (149, 40)
            mtx_latent_df.to_csv("/data_qnap/yifeis/NAS/ae/results/model_hc_results/latent/"+group+"_"+str(sub)+"_"+run+"_surf_"+prepro+"_latent_df.csv")

    print('saved test results!')

    '''
        Calculate and save regional MSE for testing data
    '''
    # calculation
    mse_region_360_2 = mse_by_region(test_data_360_2mm, test_recon_data_360_2mm)
    mse_region_360_4 = mse_by_region(test_data_360_4mm, test_recon_data_360_4mm)
    mse_region_379 = mse_by_region(test_data_379, test_recon_data_379)


    # save MSE
    for sub in test_subjects[:50]:
        # 2mm
        mse_data_sub = mse_region_360_2[sub]
        cols = list(hcp.mmp.labels.values())[1:361] # 360
        rows = test_sessions_360_2mm[sub]
        mse_data_sub = pd.DataFrame(data=mse_data_sub, index=rows, columns=cols) # (2, 360)
        mse_data_sub.to_csv("/data_qnap/yifeis/NAS/ae/results/model_hc_results/MSE/"+group+"_"+str(sub)+"_2mm_"+prepro+"_regional_MSE.csv")

        # 4mm
        mse_data_sub = mse_region_360_4[sub]
        cols = list(hcp.mmp.labels.values())[1:361] # 360
        rows = test_sessions_360_4mm[sub]
        mse_data_sub = pd.DataFrame(data=mse_data_sub, index=rows, columns=cols) # (2, 360)
        mse_data_sub.to_csv("/data_qnap/yifeis/NAS/ae/results/model_hc_results/MSE/"+group+"_"+str(sub)+"_4mm_"+prepro+"_regional_MSE.csv")

        # surface
        mse_data_sub = mse_region_379[sub]
        cols = list(hcp.mmp.labels.values())[1:] # 379
        rows = test_sessions_379[sub]
        mse_data_sub = pd.DataFrame(data=mse_data_sub, index=rows, columns=cols) # (2, 379)
        mse_data_sub.to_csv("/data_qnap/yifeis/NAS/ae/results/model_hc_results/MSE/"+group+"_"+str(sub)+"_surf_"+prepro+"_regional_MSE.csv")


    '''
        Save plots
    '''

    for sub in test_subjects[:50]:

        # volume MSE 2mm
        mse_sub = mse_region_360_2[sub]       # regional mse data of sub
        test_ses = test_sessions_360_2mm[sub]
        for i in range(len(mse_sub)):
            mse = mse_sub[i]
            run = test_ses[i]
            ## padding -1 to subcortical regions
            pad = np.ones(19) - 2
            mse = np.concatenate((mse, pad))
            # save image of mse
            nlp.view_surf(hcp.mesh.inflated,hcp.cortex_data(hcp.unparcellate(mse, hcp.mmp)), bg_map=hcp.mesh.sulc, symmetric_cmap = False, vmax = 0.01, vmin = 0).save_as_html('/data_qnap/yifeis/NAS/ae/results/model_hc_results/plots/'+group+"_"+str(sub)+"_"+run+"_2mm_"+prepro+'_mse.html')

        # volume MSE 4mm
        mse_sub = mse_region_360_4[sub]       # regional mse data of sub
        test_ses = test_sessions_360_4mm[sub]
        for i in range(len(mse_sub)):
            mse = mse_sub[i]
            run = test_ses[i]
            ## padding -1 to subcortical regions
            pad = np.ones(19) - 2
            mse = np.concatenate((mse, pad))
            # save image of mse
            nlp.view_surf(hcp.mesh.inflated,hcp.cortex_data(hcp.unparcellate(mse, hcp.mmp)), bg_map=hcp.mesh.sulc, symmetric_cmap = False, vmax = 0.01, vmin = 0).save_as_html('/data_qnap/yifeis/NAS/ae/results/model_hc_results/plots/'+group+"_"+str(sub)+"_"+run+"_4mm_"+prepro+'_mse.html')

        # surface
        mse_sub = mse_region_379[sub]       # regional mse data of sub
        test_ses = test_sessions_379[sub]
        for i in range(len(mse_sub)):
            mse = mse_sub[i]
            run = test_ses[i]
            # save image of mse
            nlp.view_surf(hcp.mesh.inflated,hcp.cortex_data(hcp.unparcellate(mse, hcp.mmp)), bg_map=hcp.mesh.sulc, symmetric_cmap = False, vmax = 0.01, vmin = 0).save_as_html('/data_qnap/yifeis/NAS/ae/results/model_hc_results/plots/'+group+"_"+str(sub)+"_"+run+"_surf_"+prepro+'_mse.html')
else:
    print("Test on HCP data")
    # test on the HCP subjects
    hcp_subjects = os.listdir('/data_qnap/yifeis/new/processed/')[50:75]
    print(len(hcp_subjects))
    print(hcp_subjects)
    hcp_data_379, hcp_data_360, hcp_session = load_HCP_data(hcp_subjects)
    '''
        Test
    '''
    test_latent_data_360 = {}
    test_recon_data_360  = {}

    test_latent_data_379 = {}
    test_recon_data_379  = {}

    for sub in hcp_subjects:
        # get the test data for each subject
        sub_360 = hcp_data_360[sub]
        sub_379 = hcp_data_379[sub]

        sub_l_360 = []
        sub_r_360 = []

        sub_l_379 = []
        sub_r_379 = []

        for mtx in sub_360: # (900, 360)
            latent, recon = test(encoder_360, decoder_360, mtx)
            sub_l_360.append(latent)
            sub_r_360.append(recon)

        for mtx in sub_379: # (900, 379)
            latent, recon = test(encoder_379, decoder_379, mtx)
            sub_l_379.append(latent)
            sub_r_379.append(recon)

        test_latent_data_360[sub] = sub_l_360
        test_recon_data_360[sub] = sub_r_360

        test_latent_data_379[sub] = sub_l_379
        test_recon_data_379[sub] = sub_r_379

    print('test finished!')

    '''
        Save reconstruction results
        Save latent variables
    '''

    for sub in hcp_subjects:
        # 360 regions
        test_re = test_recon_data_360[sub]           # reconstructed data for sub
        test_latent_re = test_latent_data_360[sub]   # latent data for sub
        test_ses_re = hcp_session[sub]               # session information for sub
        # save reconstructed/latent data
        for i in range(len(test_re)):
            run = test_ses_re[i]
            # save reconstructed data
            mtx_recon_df = pd.DataFrame(data=test_re[i]) # (900, 360)
            mtx_recon_df.to_csv("/data_qnap/yifeis/NAS/ae/results/model_hc_results/recon/HCP_"+str(sub)+"_"+run+"_360_recon_df.csv")
            # save latent data
            mtx_latent_df = pd.DataFrame(data=test_latent_re[i]) # (900, 45)
            mtx_latent_df.to_csv("/data_qnap/yifeis/NAS/ae/results/model_hc_results/latent/HCP_"+str(sub)+"_"+run+"_360_latent_df.csv")

        # 379
        test_re = test_recon_data_379[sub]           # reconstructed data for sub
        test_latent_re = test_latent_data_379[sub]   # latent data for sub
        test_ses_re = hcp_session[sub]               # session information for sub
        # save reconstructed/latent data
        for i in range(len(test_re)):
            run = test_ses_re[i]
            # save reconstructed data
            mtx_recon_df = pd.DataFrame(data=test_re[i])  # (900, 379)
            mtx_recon_df.to_csv("/data_qnap/yifeis/NAS/ae/results/model_hc_results/recon/HCP_"+str(sub)+"_"+run+"_379_recon_df.csv")
            # save latent data
            mtx_latent_df = pd.DataFrame(data=test_latent_re[i]) # (900, 40)
            mtx_latent_df.to_csv("/data_qnap/yifeis/NAS/ae/results/model_hc_results/latent/HCP_"+str(sub)+"_"+run+"_379_latent_df.csv")

    print('saved test results!')

    '''
        Calculate and save regional MSE for testing data
    '''
    # calculation
    mse_region_360 = mse_by_region(hcp_data_360, test_recon_data_360)
    mse_region_379 = mse_by_region(hcp_data_379, test_recon_data_379)

    # save MSE
    for sub in hcp_subjects:
        # 360
        mse_data_sub = mse_region_360[sub]
        cols = list(hcp.mmp.labels.values())[1:361] # 360
        rows = hcp_session[sub]
        mse_data_sub = pd.DataFrame(data=mse_data_sub, index=rows, columns=cols) # (2, 360)
        mse_data_sub.to_csv("/data_qnap/yifeis/NAS/ae/results/model_hc_results/MSE/HCP_"+str(sub)+"_360_regional_MSE.csv")

        # 379
        mse_data_sub = mse_region_379[sub]
        cols = list(hcp.mmp.labels.values())[1:] # 379
        rows = hcp_session[sub]
        mse_data_sub = pd.DataFrame(data=mse_data_sub, index=rows, columns=cols) # (2, 379)
        mse_data_sub.to_csv("/data_qnap/yifeis/NAS/ae/results/model_hc_results/MSE/HCP_"+str(sub)+"_379_regional_MSE.csv")

    '''
        Save plots
    '''

    for sub in hcp_subjects:

        # 360 regions
        mse_sub = mse_region_360[sub]       # regional mse data of sub
        test_ses = hcp_session[sub]
        for i in range(len(mse_sub)):
            mse = mse_sub[i]
            run = test_ses[i]
            ## padding -1 to subcortical regions
            pad = np.ones(19) - 2
            mse = np.concatenate((mse, pad))
            # save image of mse
            nlp.view_surf(hcp.mesh.inflated,hcp.cortex_data(hcp.unparcellate(mse, hcp.mmp)), bg_map=hcp.mesh.sulc, symmetric_cmap = False, vmax = 0.01, vmin = 0).save_as_html('/data_qnap/yifeis/NAS/ae/results/model_hc_results/plots/HCP_'+str(sub)+"_"+run+'_360_mse.html')

        # 379
        mse_sub = mse_region_379[sub]       # regional mse data of sub
        test_ses = hcp_session[sub]
        for i in range(len(mse_sub)):
            mse = mse_sub[i]
            run = test_ses[i]
            # save image of mse
            nlp.view_surf(hcp.mesh.inflated,hcp.cortex_data(hcp.unparcellate(mse, hcp.mmp)), bg_map=hcp.mesh.sulc, symmetric_cmap = False, vmax = 0.01, vmin = 0).save_as_html('/data_qnap/yifeis/NAS/ae/results/model_hc_results/plots/HCP_'+str(sub)+"_"+run+'_379_mse.html')
