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
from tensorflow.keras import losses, callbacks
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# ------ Functions ------ #
def load_data(subjects, ls):
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

        mtx1_p = mtx1_p.T
        mtx1_p = (mtx1_p - np.mean(mtx1_p,axis=0))/np.std(mtx1_p,axis=0)
        mtx1_p = mtx1_p.T

        mtx2_p = mtx2_p.T
        mtx2_p = (mtx2_p - np.mean(mtx2_p,axis=0))/np.std(mtx2_p,axis=0)
        mtx2_p = mtx2_p.T

        mtx3_p = mtx3_p.T
        mtx3_p = (mtx3_p - np.mean(mtx3_p,axis=0))/np.std(mtx3_p,axis=0)
        mtx3_p = mtx3_p.T

        mtx4_p = mtx4_p.T
        mtx4_p = (mtx4_p - np.mean(mtx4_p,axis=0))/np.std(mtx4_p,axis=0)
        mtx4_p = mtx4_p.T

        mtx5_p = mtx5_p.T
        mtx5_p = (mtx5_p - np.mean(mtx5_p,axis=0))/np.std(mtx5_p,axis=0)
        mtx5_p = mtx5_p.T

        mtx6_p = mtx6_p.T
        mtx6_p = (mtx6_p - np.mean(mtx6_p,axis=0))/np.std(mtx6_p,axis=0)
        mtx6_p = mtx6_p.T

        mtx7_p = mtx7_p.T
        mtx7_p = (mtx7_p - np.mean(mtx7_p,axis=0))/np.std(mtx7_p,axis=0)
        mtx7_p = mtx7_p.T

        mtx8_p = mtx8_p.T
        mtx8_p = (mtx8_p - np.mean(mtx8_p,axis=0))/np.std(mtx8_p,axis=0)
        mtx8_p = mtx8_p.T

        mtx9_p = mtx9_p.T
        mtx9_p = (mtx9_p - np.mean(mtx9_p,axis=0))/np.std(mtx9_p,axis=0)
        mtx9_p = mtx9_p.T

        mtx10_p = mtx10_p.T
        mtx10_p = (mtx10_p - np.mean(mtx10_p,axis=0))/np.std(mtx10_p,axis=0)
        mtx10_p = mtx10_p.T

        mtx11_p = mtx11_p.T
        mtx11_p = (mtx11_p - np.mean(mtx11_p,axis=0))/np.std(mtx11_p,axis=0)
        mtx11_p = mtx11_p.T

        mtx12_p = mtx12_p.T
        mtx12_p = (mtx12_p - np.mean(mtx12_p,axis=0))/np.std(mtx12_p,axis=0)
        mtx12_p = mtx12_p.T

        mtx13_p = mtx13_p.T
        mtx13_p = (mtx13_p - np.mean(mtx13_p,axis=0))/np.std(mtx13_p,axis=0)
        mtx13_p = mtx13_p.T

        mtx14_p = mtx14_p.T
        mtx14_p = (mtx14_p - np.mean(mtx14_p,axis=0))/np.std(mtx14_p,axis=0)
        mtx14_p = mtx14_p.T
        ls.append(mtx1_p)
        ls.append(mtx2_p)
        ls.append(mtx3_p)
        ls.append(mtx4_p)
        ls.append(mtx5_p)
        ls.append(mtx6_p)
        ls.append(mtx7_p)
        ls.append(mtx8_p)
        ls.append(mtx9_p)
        ls.append(mtx10_p)
        ls.append(mtx11_p)
        ls.append(mtx12_p)
        ls.append(mtx13_p)
        ls.append(mtx14_p)

def load_train_data(subjects, ls):
    for subject in subjects:
        mtx1_p = np.load('/data_qnap/yifeis/new/processed/'+str(subject)+'/rest1_p.npy')
        mtx2_p = np.load('/data_qnap/yifeis/new/processed/'+str(subject)+'/rest2_p.npy')
        mtx3_p = np.load('/data_qnap/yifeis/new/processed/'+str(subject)+'/rest3_p.npy')
        mtx4_p = np.load('/data_qnap/yifeis/new/processed/'+str(subject)+'/rest4_p.npy')
        mtx1_p = mtx1_p.T
        mtx1_p = (mtx1_p - np.mean(mtx1_p,axis=0))/np.std(mtx1_p,axis=0)
        mtx1_p = mtx1_p.T

        mtx2_p = mtx2_p.T
        mtx2_p = (mtx2_p - np.mean(mtx2_p,axis=0))/np.std(mtx2_p,axis=0)
        mtx2_p = mtx2_p.T

        mtx3_p = mtx3_p.T
        mtx3_p = (mtx3_p - np.mean(mtx3_p,axis=0))/np.std(mtx3_p,axis=0)
        mtx3_p = mtx3_p.T

        mtx4_p = mtx4_p.T
        mtx4_p = (mtx4_p - np.mean(mtx4_p,axis=0))/np.std(mtx4_p,axis=0)
        mtx4_p = mtx4_p.T

        ls.append(mtx1_p)
        ls.append(mtx2_p)
        ls.append(mtx3_p)
        ls.append(mtx4_p)

# ------ MSE calculation ------ #
def mse(input, output):
    difference_array = np.subtract(input, output)
    squared_array = np.square(difference_array)
    mse = squared_array.mean()
    return mse

def mse_by_region(input, output):
    mse_all = []
    # mse_func = losses.MeanSquaredError()
    for m in range(len(input)): # for each subject
        subject_i = input[m]
        subject_o = output[m]
        mse_sub = []
        for n in range(len(subject_i)):# for each session
            mtx_i = subject_i[n]
            mtx_o = subject_o[n]
            mse_ses = np.zeros(mtx_i.shape[1])
            for i in range(mtx_i.shape[1]): # for each region
                region_i =  mtx_i[:, i]
                region_o =  mtx_o[:, i]
                mse_ses[i] = mse(region_i, region_o)
            mse_sub.append(mse_ses)
        mse_all.append(mse_sub)
    return mse_all

# ------ Autoencoder ------ #
def train(train_data, epo, name):

    train_sub = []
    for n in range(len(train_subjects)):
        train_sub.append(train_data[n*4: n*4 + 4])

    # encoder
    input_data = Input(shape=(379,))
    encoder1 = Dense(160, activation='relu')(input_data)
    encoder2 = Dense(80, activation='relu')(encoder1)
    encoder3 = Dense(40, activation='relu')(encoder2)

    # decoder
    decoder1 = Dense(80, activation='relu')(encoder3)
    decoder2 = Dense(160, activation='relu')(decoder1)
    decoder3 = Dense(379)(decoder2)

    # train
    hist_ls = []
    autoencoder = Model(inputs=input_data, outputs=decoder3)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error',metrics=[losses.MeanSquaredError(), 'accuracy'])
    sub_idx = list(range(len(train_sub)))
    # round 1
    for sub_i in sub_idx: # for each subject in origional order
        sub = train_sub[sub_i]
        print("First round: training with subject " + str(train_subjects[sub_i]))
        for tr in sub: # train the model with each matrix from this subject for epo epochs
            print(tr.shape)
            h = autoencoder.fit(tr, tr, epochs=epo, batch_size=32, shuffle=True)
            hist_ls.append(h)

    # round 2
    random.shuffle(sub_idx)
    for sub_i in sub_idx: # for each subject in origional order
        sub = train_sub[sub_i]
        print("Second round: training with subject " + str(train_subjects[sub_i]))
        for tr in sub: # train the model with each matrix from this subject for epo epochs
            h = autoencoder.fit(tr, tr, epochs=epo, batch_size=32, shuffle=True)
            hist_ls.append(h)

    # round 3
    random.shuffle(sub_idx)
    for sub_i in sub_idx: # for each subject in origional order
        sub = train_sub[sub_i]
        print("Third round: training with subject " + str(train_subjects[sub_i]))
        for tr in sub: # train the model with each matrix from this subject for epo epochs
            h = autoencoder.fit(tr, tr, epochs=epo, batch_size=32, shuffle=True)
            hist_ls.append(h)

    print(autoencoder.summary())
    # create encoder model
    encoder = Model(inputs=input_data, outputs=encoder3)
    # create decoder model
    encoded_input = Input(shape=(40,))
    decoder_layer1 = autoencoder.layers[-3]
    decoder_layer2 = autoencoder.layers[-2]
    decoder_layer3 = autoencoder.layers[-1]
    decoder = Model(inputs=encoded_input, outputs=decoder_layer3(decoder_layer2(decoder_layer1(encoded_input))))

    # save the two decoder layers
    decoder_model_1 = Model(inputs=input_data, outputs=decoder1)
    decoder_model_2 = Model(inputs=input_data, outputs=decoder2)
    ## save the model
    # saving whole model
    autoencoder.save('/data_qnap/yifeis/ae_relu_trained_with_50/model/autoencoder.hdf5')
    encoder.save('/data_qnap/yifeis/ae_relu_trained_with_50/model/encoder.hdf5')
    decoder.save('/data_qnap/yifeis/ae_relu_trained_with_50/model/decoder.hdf5')
    decoder_model_1.save('/data_qnap/yifeis/ae_relu_trained_with_50/model/decoder_80.hdf5')
    decoder_model_2.save('/data_qnap/yifeis/ae_relu_trained_with_50/model/decoder_160.hdf5')
    print("Saving Successful!")

    ## total history
    # create an empty dict to save all three history dicts into
    total_history_dict = dict()
    for some_key in hist_ls[0].history.keys():
        current_values = [] # to save values from all three hist dicts
        for hist_dict in hist_ls:
            hist_dict = hist_dict.history
            current_values += hist_dict[some_key]
        total_history_dict[some_key] = current_values
    return (autoencoder, encoder, decoder, total_history_dict, decoder_model_1, decoder_model_2)

def test(encoder, decoder, decoder_80, decoder_160, test_data):
    # latent_vector
    latent_vector = encoder.predict(test_data)
    # decoder output
    reconstructed_data = decoder.predict(latent_vector)
    # decoder 80 nodes output
    recon_80 = decoder_80.predict(test_data)
    # decoder 160 nodes output
    recon_160 = decoder_160.predict(test_data)
    return (latent_vector, reconstructed_data, recon_80, recon_160)

# ------ Load Data ------ #
# 50
subjects  = [100610, 102311, 102816, 104416, 105923,
             108323, 109123, 111514, 114823, 115017,
             115825, 116726, 118225, 125525, 126426,
             128935, 130114, 130518, 131217, 131722,
             132118, 134627, 134829, 135124, 137128,
             140117, 144226, 145834, 146129, 146432,
             146735, 146937, 148133, 150423, 155938,
             156334, 157336, 158035, 158136, 159239,
             162935, 164131, 164636, 165436, 167036,
             167440, 169040, 169343, 169444, 169747]

# load data
test_subjects  = subjects
train_subjects = subjects

train_ls = []
test_ls  = []
load_train_data(train_subjects, train_ls)
load_data(test_subjects, test_ls)

# train
epochs = 2
autoencoder, encoder, decoder, total_history,decoder_model_1, decoder_model_2  = train(train_ls, epochs, 'basic_ae')

# test
test_latent_ls = []
test_recon_ls  = []
test_80_ls = []
test_160_ls  = []
for t in test_ls:
    latent, recon, recon_80, recon_160 = test(encoder, decoder,decoder_model_1, decoder_model_2, t)
    test_latent_ls.append(latent)
    test_recon_ls.append(recon)
    test_80_ls.append(recon_80)
    test_160_ls.append(recon_160)

# Separate the results for different subjects
test_recon_sub  = []
test_latent_sub = []
test_80_sub  = []
test_160_sub = []
test_sub  = []
for n in range(len(test_subjects)):
    test_recon_sub.append(test_recon_ls[n*14: n*14 + 14])
    test_latent_sub.append(test_latent_ls[n*14: n*14 + 14])
    test_80_sub.append(test_80_ls[n*14: n*14 + 14])
    test_160_sub.append(test_160_ls[n*14: n*14 + 14])
    test_sub.append(test_ls[n*14: n*14 + 14])

'''
    Save reconstruction results
    Save latent variables
'''
for sub_idx in range(len(test_recon_sub)):
    sub = test_subjects[sub_idx]
    test_re = test_recon_sub[sub_idx]           # reconstructed data for sub
    test_latent_re = test_latent_sub[sub_idx]   # latent data for sub
    test_80_re = test_80_sub[sub_idx]           # reconstructed data for sub
    test_160_re = test_160_sub[sub_idx]
    # save reconstructed data
    m1_recon_df = pd.DataFrame(data=test_re[0])
    m2_recon_df = pd.DataFrame(data=test_re[1])
    m3_recon_df = pd.DataFrame(data=test_re[2])
    m4_recon_df = pd.DataFrame(data=test_re[3])

    r1_recon_df = pd.DataFrame(data=test_re[4])
    r2_recon_df = pd.DataFrame(data=test_re[5])
    r3_recon_df = pd.DataFrame(data=test_re[6])
    r4_recon_df = pd.DataFrame(data=test_re[7])

    t1_recon_df = pd.DataFrame(data=test_re[8])
    t2_recon_df = pd.DataFrame(data=test_re[9])
    t3_recon_df = pd.DataFrame(data=test_re[10])
    t4_recon_df = pd.DataFrame(data=test_re[11])
    t5_recon_df = pd.DataFrame(data=test_re[12])
    t6_recon_df = pd.DataFrame(data=test_re[13])

    m1_recon_df.to_csv("/data_qnap/yifeis/ae_relu_trained_with_50/recon/"+str(sub)+"_movie1_recon_df.csv")
    m2_recon_df.to_csv("/data_qnap/yifeis/ae_relu_trained_with_50/recon/"+str(sub)+"_movie2_recon_df.csv")
    m3_recon_df.to_csv("/data_qnap/yifeis/ae_relu_trained_with_50/recon/"+str(sub)+"_movie3_recon_df.csv")
    m4_recon_df.to_csv("/data_qnap/yifeis/ae_relu_trained_with_50/recon/"+str(sub)+"_movie4_recon_df.csv")

    r1_recon_df.to_csv("/data_qnap/yifeis/ae_relu_trained_with_50/recon/"+str(sub)+"_rest1_recon_df.csv")
    r2_recon_df.to_csv("/data_qnap/yifeis/ae_relu_trained_with_50/recon/"+str(sub)+"_rest2_recon_df.csv")
    r3_recon_df.to_csv("/data_qnap/yifeis/ae_relu_trained_with_50/recon/"+str(sub)+"_rest3_recon_df.csv")
    r4_recon_df.to_csv("/data_qnap/yifeis/ae_relu_trained_with_50/recon/"+str(sub)+"_rest4_recon_df.csv")

    t1_recon_df.to_csv("/data_qnap/yifeis/ae_relu_trained_with_50/recon/"+str(sub)+"_retbar1_recon_df.csv")
    t2_recon_df.to_csv("/data_qnap/yifeis/ae_relu_trained_with_50/recon/"+str(sub)+"_retbar2_recon_df.csv")
    t3_recon_df.to_csv("/data_qnap/yifeis/ae_relu_trained_with_50/recon/"+str(sub)+"_retccw_recon_df.csv")
    t4_recon_df.to_csv("/data_qnap/yifeis/ae_relu_trained_with_50/recon/"+str(sub)+"_retcon_recon_df.csv")
    t5_recon_df.to_csv("/data_qnap/yifeis/ae_relu_trained_with_50/recon/"+str(sub)+"_retcw_recon_df.csv")
    t6_recon_df.to_csv("/data_qnap/yifeis/ae_relu_trained_with_50/recon/"+str(sub)+"_retexp_recon_df.csv")

    # save latent data
    m1_latent_df = pd.DataFrame(data=test_latent_re[0])
    m2_latent_df = pd.DataFrame(data=test_latent_re[1])
    m3_latent_df = pd.DataFrame(data=test_latent_re[2])
    m4_latent_df = pd.DataFrame(data=test_latent_re[3])

    r1_latent_df = pd.DataFrame(data=test_latent_re[4])
    r2_latent_df = pd.DataFrame(data=test_latent_re[5])
    r3_latent_df = pd.DataFrame(data=test_latent_re[6])
    r4_latent_df = pd.DataFrame(data=test_latent_re[7])

    t1_latent_df = pd.DataFrame(data=test_latent_re[8])
    t2_latent_df = pd.DataFrame(data=test_latent_re[9])
    t3_latent_df = pd.DataFrame(data=test_latent_re[10])
    t4_latent_df = pd.DataFrame(data=test_latent_re[11])
    t5_latent_df = pd.DataFrame(data=test_latent_re[12])
    t6_latent_df = pd.DataFrame(data=test_latent_re[13])

    m1_latent_df.to_csv("/data_qnap/yifeis/ae_relu_trained_with_50/latent/"+str(sub)+"_movie1_latent_df.csv")
    m2_latent_df.to_csv("/data_qnap/yifeis/ae_relu_trained_with_50/latent/"+str(sub)+"_movie2_latent_df.csv")
    m3_latent_df.to_csv("/data_qnap/yifeis/ae_relu_trained_with_50/latent/"+str(sub)+"_movie3_latent_df.csv")
    m4_latent_df.to_csv("/data_qnap/yifeis/ae_relu_trained_with_50/latent/"+str(sub)+"_movie4_latent_df.csv")

    r1_latent_df.to_csv("/data_qnap/yifeis/ae_relu_trained_with_50/latent/"+str(sub)+"_rest1_latent_df.csv")
    r2_latent_df.to_csv("/data_qnap/yifeis/ae_relu_trained_with_50/latent/"+str(sub)+"_rest2_latent_df.csv")
    r3_latent_df.to_csv("/data_qnap/yifeis/ae_relu_trained_with_50/latent/"+str(sub)+"_rest3_latent_df.csv")
    r4_latent_df.to_csv("/data_qnap/yifeis/ae_relu_trained_with_50/latent/"+str(sub)+"_rest4_latent_df.csv")

    t1_latent_df.to_csv("/data_qnap/yifeis/ae_relu_trained_with_50/latent/"+str(sub)+"_retbar1_latent_df.csv")
    t2_latent_df.to_csv("/data_qnap/yifeis/ae_relu_trained_with_50/latent/"+str(sub)+"_retbar2_latent_df.csv")
    t3_latent_df.to_csv("/data_qnap/yifeis/ae_relu_trained_with_50/latent/"+str(sub)+"_retccw_latent_df.csv")
    t4_latent_df.to_csv("/data_qnap/yifeis/ae_relu_trained_with_50/latent/"+str(sub)+"_retcon_latent_df.csv")
    t5_latent_df.to_csv("/data_qnap/yifeis/ae_relu_trained_with_50/latent/"+str(sub)+"_retcw_latent_df.csv")
    t6_latent_df.to_csv("/data_qnap/yifeis/ae_relu_trained_with_50/latent/"+str(sub)+"_retexp_latent_df.csv")

    # save 80 layer data
    m1_recon_80_df = pd.DataFrame(data=test_80_re[0])
    m2_recon_80_df = pd.DataFrame(data=test_80_re[1])
    m3_recon_80_df = pd.DataFrame(data=test_80_re[2])
    m4_recon_80_df = pd.DataFrame(data=test_80_re[3])

    r1_recon_80_df = pd.DataFrame(data=test_80_re[4])
    r2_recon_80_df = pd.DataFrame(data=test_80_re[5])
    r3_recon_80_df = pd.DataFrame(data=test_80_re[6])
    r4_recon_80_df = pd.DataFrame(data=test_80_re[7])

    t1_recon_80_df = pd.DataFrame(data=test_80_re[8])
    t2_recon_80_df = pd.DataFrame(data=test_80_re[9])
    t3_recon_80_df = pd.DataFrame(data=test_80_re[10])
    t4_recon_80_df = pd.DataFrame(data=test_80_re[11])
    t5_recon_80_df = pd.DataFrame(data=test_80_re[12])
    t6_recon_80_df = pd.DataFrame(data=test_80_re[13])

    m1_recon_80_df.to_csv("/data_qnap/yifeis/ae_relu_trained_with_50/recon_80/"+str(sub)+"_movie1_recon_80_df.csv")
    m2_recon_80_df.to_csv("/data_qnap/yifeis/ae_relu_trained_with_50/recon_80/"+str(sub)+"_movie2_recon_80_df.csv")
    m3_recon_80_df.to_csv("/data_qnap/yifeis/ae_relu_trained_with_50/recon_80/"+str(sub)+"_movie3_recon_80_df.csv")
    m4_recon_80_df.to_csv("/data_qnap/yifeis/ae_relu_trained_with_50/recon_80/"+str(sub)+"_movie4_recon_80_df.csv")

    r1_recon_80_df.to_csv("/data_qnap/yifeis/ae_relu_trained_with_50/recon_80/"+str(sub)+"_rest1_recon_80_df.csv")
    r2_recon_80_df.to_csv("/data_qnap/yifeis/ae_relu_trained_with_50/recon_80/"+str(sub)+"_rest2_recon_80_df.csv")
    r3_recon_80_df.to_csv("/data_qnap/yifeis/ae_relu_trained_with_50/recon_80/"+str(sub)+"_rest3_recon_80_df.csv")
    r4_recon_80_df.to_csv("/data_qnap/yifeis/ae_relu_trained_with_50/recon_80/"+str(sub)+"_rest4_recon_80_df.csv")

    t1_recon_80_df.to_csv("/data_qnap/yifeis/ae_relu_trained_with_50/recon_80/"+str(sub)+"_retbar1_recon_80_df.csv")
    t2_recon_80_df.to_csv("/data_qnap/yifeis/ae_relu_trained_with_50/recon_80/"+str(sub)+"_retbar2_recon_80_df.csv")
    t3_recon_80_df.to_csv("/data_qnap/yifeis/ae_relu_trained_with_50/recon_80/"+str(sub)+"_retccw_recon_80_df.csv")
    t4_recon_80_df.to_csv("/data_qnap/yifeis/ae_relu_trained_with_50/recon_80/"+str(sub)+"_retcon_recon_80_df.csv")
    t5_recon_80_df.to_csv("/data_qnap/yifeis/ae_relu_trained_with_50/recon_80/"+str(sub)+"_retcw_recon_80_df.csv")
    t6_recon_80_df.to_csv("/data_qnap/yifeis/ae_relu_trained_with_50/recon_80/"+str(sub)+"_retexp_recon_80_df.csv")

    # save 160 layer data
    m1_recon_160_df = pd.DataFrame(data=test_160_re[0])
    m2_recon_160_df = pd.DataFrame(data=test_160_re[1])
    m3_recon_160_df = pd.DataFrame(data=test_160_re[2])
    m4_recon_160_df = pd.DataFrame(data=test_160_re[3])

    r1_recon_160_df = pd.DataFrame(data=test_160_re[4])
    r2_recon_160_df = pd.DataFrame(data=test_160_re[5])
    r3_recon_160_df = pd.DataFrame(data=test_160_re[6])
    r4_recon_160_df = pd.DataFrame(data=test_160_re[7])

    t1_recon_160_df = pd.DataFrame(data=test_160_re[8])
    t2_recon_160_df = pd.DataFrame(data=test_160_re[9])
    t3_recon_160_df = pd.DataFrame(data=test_160_re[10])
    t4_recon_160_df = pd.DataFrame(data=test_160_re[11])
    t5_recon_160_df = pd.DataFrame(data=test_160_re[12])
    t6_recon_160_df = pd.DataFrame(data=test_160_re[13])

    m1_recon_160_df.to_csv("/data_qnap/yifeis/ae_relu_trained_with_50/recon_160/"+str(sub)+"_movie1_recon_160_df.csv")
    m2_recon_160_df.to_csv("/data_qnap/yifeis/ae_relu_trained_with_50/recon_160/"+str(sub)+"_movie2_recon_160_df.csv")
    m3_recon_160_df.to_csv("/data_qnap/yifeis/ae_relu_trained_with_50/recon_160/"+str(sub)+"_movie3_recon_160_df.csv")
    m4_recon_160_df.to_csv("/data_qnap/yifeis/ae_relu_trained_with_50/recon_160/"+str(sub)+"_movie4_recon_160_df.csv")

    r1_recon_160_df.to_csv("/data_qnap/yifeis/ae_relu_trained_with_50/recon_160/"+str(sub)+"_rest1_recon_160_df.csv")
    r2_recon_160_df.to_csv("/data_qnap/yifeis/ae_relu_trained_with_50/recon_160/"+str(sub)+"_rest2_recon_160_df.csv")
    r3_recon_160_df.to_csv("/data_qnap/yifeis/ae_relu_trained_with_50/recon_160/"+str(sub)+"_rest3_recon_160_df.csv")
    r4_recon_160_df.to_csv("/data_qnap/yifeis/ae_relu_trained_with_50/recon_160/"+str(sub)+"_rest4_recon_160_df.csv")

    t1_recon_160_df.to_csv("/data_qnap/yifeis/ae_relu_trained_with_50/recon_160/"+str(sub)+"_retbar1_recon_160_df.csv")
    t2_recon_160_df.to_csv("/data_qnap/yifeis/ae_relu_trained_with_50/recon_160/"+str(sub)+"_retbar2_recon_160_df.csv")
    t3_recon_160_df.to_csv("/data_qnap/yifeis/ae_relu_trained_with_50/recon_160/"+str(sub)+"_retccw_recon_160_df.csv")
    t4_recon_160_df.to_csv("/data_qnap/yifeis/ae_relu_trained_with_50/recon_160/"+str(sub)+"_retcon_recon_160_df.csv")
    t5_recon_160_df.to_csv("/data_qnap/yifeis/ae_relu_trained_with_50/recon_160/"+str(sub)+"_retcw_recon_160_df.csv")
    t6_recon_160_df.to_csv("/data_qnap/yifeis/ae_relu_trained_with_50/recon_160/"+str(sub)+"_retexp_recon_160_df.csv")
'''
    Calculate and save regional MSE for testing data
'''
# calculation
mse_region = mse_by_region(test_sub, test_recon_sub)

# save MSE
for i in range(len(mse_region)):
    sub = mse_region[i]
    subject = test_subjects[i]
    cols = list(hcp.mmp.labels.values())[1:]
    rows = ["movie1", "movie2", "movie3", "movie4", "rest1", "rest2", "rest3", "rest4",
            "retbar1", "retbar2", "retccw", "retcon", "retcw", "retexp"]
    sub = pd.DataFrame(data=sub, index=rows, columns=cols)
    sub.to_csv("/data_qnap/yifeis/ae_relu_trained_with_50/MSE/"+str(subject)+"_regional_MSE.csv")

# history
# print(total_history.keys())
plt.plot(total_history["mean_squared_error"])
plt.xlabel("# of iterations")
plt.ylabel("MSE")
plt.title('Train_MSE')
plt.savefig('/data_qnap/yifeis/ae_relu_trained_with_50/model/train_MSE.png')
plt.clf()
