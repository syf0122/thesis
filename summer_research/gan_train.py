import os
import sys
import pathlib
import shutil
import tempfile
import random
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import nibabel as nib
import hcp_utils as hcp
from nilearn import plotting as nlp
from nilearn import surface as nsf
from nilearn.connectome import ConnectivityMeasure
import tensorflow as tf
from tensorflow.keras import losses, callbacks, layers
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
tf.config.experimental_run_functions_eagerly(True)
# Load Training Data
# Default: HCP rs-fMRI
# For surface data: 379 regions
# For volumetric data: 360 regions (without subcortical regions)
def load_HCP_data_360(subjects, session = "rest"):
    dir = '/data_qnap/yifeis/new/processed/'
    sessions = {}
    data = {}
    for subject in subjects:
        sub_data = []
        sub_session = []
        sub_dir = dir+subject+"/"
        processed_files = os.listdir(sub_dir)
        processed_files.sort()
        for f in processed_files:
            if session in f and "_p.npy" in f:
                mtx_p = np.load(sub_dir+f)[:, :-19] # (900, 360)
                # normalize in the spatial domain
                mtx_p = mtx_p.T
                mtx_p = (mtx_p - np.mean(mtx_p,axis=0))/np.std(mtx_p,axis=0)
                mtx_p = mtx_p.T
                sub_data.append(mtx_p)
                sub_session.append(f[:-6])
        data[subject] = sub_data
        sessions[subject] = sub_session
    return (data, sessions)

def load_HCP_data_379(subjects, session = "rest"):
    dir = '/data_qnap/yifeis/new/processed/'
    sessions = {}
    data = {}
    for subject in subjects:
        sub_data = []
        sub_session = []
        sub_dir = dir+subject+"/"
        processed_files = os.listdir(sub_dir)
        processed_files.sort()
        for f in processed_files:
            if session in f and "_p.npy" in f:
                mtx_p = np.load(sub_dir+f) # (900, 379)
                # normalize in the spatial domain
                mtx_p = mtx_p.T
                mtx_p = (mtx_p - np.mean(mtx_p,axis=0))/np.std(mtx_p,axis=0)
                mtx_p = mtx_p.T
                sub_data.append(mtx_p)
                sub_session.append(f[:-6])
        data[subject] = sub_data
        sessions[subject] = sub_session
    return (data, sessions)

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
                # normalize in the spatial domain
                mtx_p = mtx_p.T
                mtx_p = (mtx_p - np.mean(mtx_p,axis=0))/np.std(mtx_p,axis=0)
                mtx_p = mtx_p.T
                sub_data.append(mtx_p)
                sub_session.append(f[13:18])
        data[subject] = sub_data
        sessions[subject] = sub_session
    return (data, sessions)

# Generator
# Autoencoder
def autoencoder(num_regions=360):
    # encoder
    input_data = Input(shape=(num_regions,))
    encoder1 = Dense(160, activation='relu')(input_data)
    encoder2 = Dense(80, activation='relu')(encoder1)
    encoder3 = Dense(40, activation='relu')(encoder2)
    # decoder
    decoder1 = Dense(80, activation='relu')(encoder3)
    decoder2 = Dense(160, activation='relu')(decoder1)
    decoder3 = Dense(num_regions)(decoder2)
    autoencoder = Model(inputs=input_data, outputs=decoder3)
    return autoencoder

# Discriminator
# CNN-based image classifier
def discriminator(num_regions=360):
    model = tf.keras.Sequential()
    model.add(layers.Dense(num_regions))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(180))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(1))
    return model

# Loss Functions
loss_mse = tf.keras.losses.MeanSquaredError()
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# loss for Discriminator
def di_loss(hcp_output, oas_output):
    hcp_loss = cross_entropy(tf.ones_like(hcp_output), hcp_output)
    oas_loss = cross_entropy(tf.zeros_like(oas_output), oas_output)
    total_loss = hcp_loss + oas_loss
    return total_loss

# loss for autoencoderd
def ae_inv_loss(oas_output):
    return cross_entropy(tf.ones_like(oas_output), oas_output)

def ae_recon_loss(h_real_data, h_regenerated_output, o_real_data, o_regenerated_output):
    h_real_data = tf.cast(h_real_data, tf.float32)
    o_real_data = tf.cast(o_real_data, tf.float32)
    h_regenerated_output = tf.cast(h_regenerated_output, tf.float32)
    o_regenerated_output = tf.cast(o_regenerated_output, tf.float32)

    total_real_data = tf.concat([h_real_data, o_real_data], 0)
    total_regenerated_data = tf.concat([h_regenerated_output, o_regenerated_output], 0)
    total_loss = loss_mse(total_real_data, total_regenerated_data)
    # h_loss = loss_mse(h_real_data, h_regenerated_output)
    # o_loss = loss_mse(o_real_data, o_regenerated_output)
    # total_loss = h_loss + o_loss
    return total_loss

# Optimizers
a_optimizer = tf.keras.optimizers.Adam(1e-4)
d_optimizer = tf.keras.optimizers.Adam(1e-4)

# Train
@tf.function
def train_step(ae_model, disc_model, hcp_mtx, oas_mtx, l, a_loss_ls, d_loss_ls, i_loss_ls):
    # GAN
    with tf.GradientTape() as ae_tape, tf.GradientTape() as disc_tape:
        # reconstruct the image
        h_recon = ae_model(hcp_mtx, training=True) # 900, regions
        o_recon = ae_model(oas_mtx, training=True) # 149, regions
        # discriminator outputs
        h_class = disc_model(h_recon, training=True)
        o_class = disc_model(o_recon, training=True)

        # calculate losses
        inv_loss   = ae_inv_loss(o_class) # inverse loss
        recon_loss = ae_recon_loss(hcp_mtx, h_recon, oas_mtx, o_recon) # recon loss
        ae_loss    = recon_loss + (1 - l) * inv_loss # total ae_loss
        disc_loss  = l * di_loss(h_class, o_class) # discriminator loss
        # record the loss
        i_loss_ls.append(inv_loss.numpy())
        a_loss_ls.append(recon_loss.numpy())
        d_loss_ls.append(disc_loss.numpy())
    gradients_of_ae   = ae_tape.gradient(ae_loss, ae_model.trainable_variables)
    gradients_of_disc = disc_tape.gradient(disc_loss, disc_model.trainable_variables)
    a_optimizer.apply_gradients(zip(gradients_of_ae, ae_model.trainable_variables))
    d_optimizer.apply_gradients(zip(gradients_of_disc, disc_model.trainable_variables))

    # # autoencoder recontstruction
    # with tf.GradientTape() as ae_tape, tf.GradientTape() as disc_tape:
    #     # reconstruct the image
    #     h_recon = ae_model(h_mtx, training=True) # 900, regions
    #     o_recon = ae_model(o_mtx, training=True) # 149, regions
    #     # calculate losses
    #     ae_loss = ae_recon_loss(h_mtx, h_recon, o_mtx, o_recon)
    #     # record the loss
    #     a_loss_ls.append(ae_loss.numpy())
    # gradients_of_ae = ae_tape.gradient(ae_loss, ae_model.trainable_variables)
    # a_optimizer.apply_gradients(zip(gradients_of_ae, ae_model.trainable_variables))

def train(hcp_train_data, oas_train_data, epo, num_regions, l, type):
    # prepare models and list to record loss history
    autoencoder_model = autoencoder(num_regions)
    discriminator_model = discriminator(num_regions)
    hcp_sub_idx = list(hcp_train_data.keys())
    oas_sub_idx = list(oas_train_data.keys())
    a_loss_ls = []
    d_loss_ls = []
    i_loss_ls = []
    # round 1
    for i in range(len(hcp_sub_idx)): # for each subject in origional order
        hcp_sub   = hcp_sub_idx[i]
        oas_sub_1 = oas_sub_idx[2*i]
        oas_sub_2 = oas_sub_idx[2*i+1]
        print("First round: training with HCP subject {} and OAS subject {} and {}.".format(hcp_sub, oas_sub_1, oas_sub_2))
        h_train_data = []
        o_train_data = []
        for mtx in hcp_train_data[hcp_sub]:
            h_train_data.append(mtx)
        for mtx in oas_train_data[oas_sub_1]:
            o_train_data.append(mtx)
        for mtx in oas_train_data[oas_sub_2]:
            o_train_data.append(mtx)
        # check if the number of scans from HCP and OAS are the same
        if len(h_train_data) != len(o_train_data):
            print("Not equal number of session from HCP and OAS data")
            print("Subjects are {}, {}, and {}".format(hcp_sub, oas_sub_1, oas_sub_2))
        progress_bar = tqdm(total = len(h_train_data) * epo)
        for i in range(len(h_train_data)): # train the model with each matrix from this subject for epo epochs
            h_mtx = h_train_data[i]
            o_mtx = o_train_data[i]
            for i in range(epo):
                train_step(autoencoder_model, discriminator_model, h_mtx, o_mtx, l, a_loss_ls, d_loss_ls, i_loss_ls)
                progress_bar.update(1)
        progress_bar.close()

    # round 2
    random.shuffle(hcp_sub_idx)
    random.shuffle(oas_sub_idx)
    for i in range(len(hcp_sub_idx)): # for each subject in origional order
        hcp_sub   = hcp_sub_idx[i]
        oas_sub_1 = oas_sub_idx[2*i]
        oas_sub_2 = oas_sub_idx[2*i+1]
        print("Second round: training with HCP subject {} and OAS subject {} and {}.".format(hcp_sub, oas_sub_1, oas_sub_2))
        h_train_data = []
        o_train_data = []
        for mtx in hcp_train_data[hcp_sub]:
            h_train_data.append(mtx)
        for mtx in oas_train_data[oas_sub_1]:
            o_train_data.append(mtx)
        for mtx in oas_train_data[oas_sub_2]:
            o_train_data.append(mtx)
        # check if the number of scans from HCP and OAS are the same
        if len(h_train_data) != len(o_train_data):
            print("Not equal number of session from HCP and OAS data")
            print("Subjects are {}, {}, and {}".format(hcp_sub, oas_sub_1, oas_sub_2))
        progress_bar = tqdm(total = len(h_train_data) * epo)
        for i in range(len(h_train_data)): # train the model with each matrix from this subject for epo epochs
            h_mtx = h_train_data[i]
            o_mtx = o_train_data[i]
            for i in range(epo):
                train_step(autoencoder_model, discriminator_model, h_mtx, o_mtx, l, a_loss_ls, d_loss_ls, i_loss_ls)
                progress_bar.update(1)
        progress_bar.close()

    # round 3
    random.shuffle(hcp_sub_idx)
    random.shuffle(oas_sub_idx)
    for i in range(len(hcp_sub_idx)): # for each subject in origional order
        hcp_sub   = hcp_sub_idx[i]
        oas_sub_1 = oas_sub_idx[2*i]
        oas_sub_2 = oas_sub_idx[2*i+1]
        print("Third round: training with HCP subject {} and OAS subject {} and {}.".format(hcp_sub, oas_sub_1, oas_sub_2))
        h_train_data = []
        o_train_data = []
        for mtx in hcp_train_data[hcp_sub]:
            h_train_data.append(mtx)
        for mtx in oas_train_data[oas_sub_1]:
            o_train_data.append(mtx)
        for mtx in oas_train_data[oas_sub_2]:
            o_train_data.append(mtx)
        # check if the number of scans from HCP and OAS are the same
        if len(h_train_data) != len(o_train_data):
            print("Not equal number of session from HCP and OAS data")
            print("Subjects are {}, {}, and {}".format(hcp_sub, oas_sub_1, oas_sub_2))
        progress_bar = tqdm(total = len(h_train_data) * epo)
        for i in range(len(h_train_data)): # train the model with each matrix from this subject for epo epochs
            h_mtx = h_train_data[i]
            o_mtx = o_train_data[i]
            for i in range(epo):
                train_step(autoencoder_model, discriminator_model, h_mtx, o_mtx, l, a_loss_ls, d_loss_ls, i_loss_ls)
                progress_bar.update(1)
        progress_bar.close()

    print("------Training Finished------")
    # saving whole model
    autoencoder_model.save('/data_qnap/yifeis/NAS/gan/with_param/models/autoencoder_'+type+'_'+str(round(l,2))+'_'+str(num_regions)+'.hdf5')
    discriminator_model.save('/data_qnap/yifeis/NAS/gan/with_param/models/discriminator_'+type+'_'+str(round(l,2))+'_'+str(num_regions)+'.hdf5')
    print("------Saved Models------")
    # save loss
    a_loss_np = np.array(a_loss_ls)
    d_loss_np = np.array(d_loss_ls)
    i_loss_np = np.array(i_loss_ls)
    a_loss_df = pd.DataFrame(a_loss_np)
    d_loss_df = pd.DataFrame(d_loss_np)
    i_loss_df = pd.DataFrame(i_loss_np)
    a_loss_df.to_csv('/data_qnap/yifeis/NAS/gan/with_param/losses/autoencoder_loss_'+type+'_'+str(round(l,2))+'_'+str(num_regions)+'.csv', index=False)
    d_loss_df.to_csv('/data_qnap/yifeis/NAS/gan/with_param/losses/discriminator_loss_'+type+'_'+str(round(l,2))+'_'+str(num_regions)+'.csv', index=False)
    i_loss_df.to_csv('/data_qnap/yifeis/NAS/gan/with_param/losses/inverse_loss_'+type+'_'+str(round(l,2))+'_'+str(num_regions)+'.csv', index=False)
    print("------Saved Loss------")
    return (autoencoder, discriminator, a_loss_np, d_loss_np, i_loss_np)


# Load Data
# Train HCP subjects
hcp_train_subjects = os.listdir("/data_qnap/yifeis/new/processed/")
hcp_train_subjects.sort()
# Train OAS subjects
oas_train_subjects = os.listdir("/data_qnap/yifeis/NAS/data/HC/")
oas_train_subjects.sort()


 # Load HCP Data
hcp_train_data_360, hcp_train_sessions_360 = load_HCP_data_360(hcp_train_subjects[:25])
hcp_train_data_379, hcp_train_sessions_379 = load_HCP_data_379(hcp_train_subjects[:25])

 # load OASIS data
oas_train_data_360_2, oas_train_sessions_360_2 = load_OAS_data("HC", oas_train_subjects[50:100], "2mm", "norm")
oas_train_data_360_4, oas_train_sessions_360_4 = load_OAS_data("HC", oas_train_subjects[50:100], "4mm", "norm")
oas_train_data_379, oas_train_sessions_379 = load_OAS_data("HC", oas_train_subjects[50:100], 'surf', "norm")

# train
lambda_value = 0.1
for n in range(89):
    lambda_value += 0.01
    lambda_value = round(lambda_value, 2)
    epochs = 160
    # autoencoder_360_2, discriminator_360_2, ae_loss_360_2, disc_loss_360_2, inv_loss_360_2 = train(hcp_train_data_360, oas_train_data_360_2, epochs, 360, lambda_value, '2mm')
    # autoencoder_360_4, discriminator_360_4, ae_loss_360_4, disc_loss_360_4, inv_loss_360_4 = train(hcp_train_data_360, oas_train_data_360_4, epochs, 360, lambda_value, '4mm')
    autoencoder_379, discriminator_379, ae_loss_379, disc_loss_379, inv_loss_379 = train(hcp_train_data_379, oas_train_data_379, epochs, 379, lambda_value, 'surf')

    #    Plot the history to verify they converged
    plot1 = plt.figure(1)
    # plt.plot(ae_loss_360_2, label="360 Regions 2mm")
    # plt.plot(ae_loss_360_4, label="360 Regions 4mm")
    plt.plot(ae_loss_379, label="379 Regions")
    plt.xlabel("# of iterations")
    plt.ylabel("MSE")
    plt.title('Loss of Autoencoders, Lambda = {}'.format(lambda_value))
    plt.legend()
    plt.savefig('/data_qnap/yifeis/NAS/gan/with_param/losses/plot_autoencoder_loss_'+str(lambda_value)+'.png')
    plt.clf()

    plot2 = plt.figure(2)
    # plt.plot(disc_loss_360_2, label="360 Regions")
    # plt.plot(disc_loss_360_4, label="360 Regions")
    plt.plot(disc_loss_379, label="379 Regions")
    plt.xlabel("# of iterations")
    plt.ylabel("Binary Cross Entropy")
    plt.title('Loss of Discriminator, Lambda = {}'.format(lambda_value))
    plt.legend()
    plt.savefig('/data_qnap/yifeis/NAS/gan/with_param/losses/plot_discriminator_loss_'+str(lambda_value)+'.png')
    plt.clf()

    plot3 = plt.figure(3)
    # plt.plot(inv_loss_360_2, label="360 Regions")
    # plt.plot(inv_loss_360_4, label="360 Regions")
    plt.plot(inv_loss_379, label="379 Regions")
    plt.xlabel("# of iterations")
    plt.ylabel("Binary Cross Entropy")
    plt.title('Inverse Loss for the Autoencoder, Lambda = {}'.format(lambda_value))
    plt.legend()
    plt.savefig('/data_qnap/yifeis/NAS/gan/with_param/losses/plot_inverse_loss_'+str(lambda_value)+'.png')
    plt.clf()
