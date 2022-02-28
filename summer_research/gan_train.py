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
                mtx_p = (mtx_p - mtx_p.min()) / (mtx_p.max() - mtx_p.min())
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
                mtx_p = (mtx_p - mtx_p.min()) / (mtx_p.max() - mtx_p.min())
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
                mtx_p = (mtx_p - mtx_p.min()) / (mtx_p.max() - mtx_p.min()) # normalize to 0-1
                sub_data.append(mtx_p)
                sub_session.append(f[13:18])
        data[subject] = sub_data
        sessions[subject] = sub_session
    return (data, sessions)

# Generator
# Autoencoder
def autoencoder(num_regions=360):
    if num_regions == 379:
        # encoder
        input_data = Input(shape=(num_regions,))
        encoder1 = Dense(160, activation='sigmoid')(input_data)
        encoder2 = Dense(80, activation='sigmoid')(encoder1)
        encoder3 = Dense(40, activation='sigmoid')(encoder2)
        # decoder
        decoder1 = Dense(80, activation='sigmoid')(encoder3)
        decoder2 = Dense(160, activation='sigmoid')(decoder1)
        decoder3 = Dense(num_regions, activation='sigmoid')(decoder2)
        autoencoder = Model(inputs=input_data, outputs=decoder3)
    else:
        # encoder
        input_data = Input(shape=(num_regions,))
        encoder1 = Dense(180, activation='sigmoid')(input_data)
        encoder2 = Dense(90, activation='sigmoid')(encoder1)
        encoder3 = Dense(45, activation='sigmoid')(encoder2)
        # decoder
        decoder1 = Dense(90, activation='sigmoid')(encoder3)
        decoder2 = Dense(180, activation='sigmoid')(decoder1)
        decoder3 = Dense(num_regions, activation='sigmoid')(decoder2)
        autoencoder = Model(inputs=input_data, outputs=decoder3)
    return autoencoder

# Discriminator
# CNN-based image classifier
def discriminator(num_regions=360):
    model = tf.keras.Sequential()
    model.add(layers.Conv1D(80, 5, input_shape=[149, num_regions])) # 149 timepoints
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv1D(40, 5))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

# Loss Functions
loss_mse = tf.keras.losses.MeanSquaredError()
# loss for Discriminator
def di_loss(hcp_output, oas_output):
    hcp_loss = loss_mse(tf.ones_like(hcp_output), hcp_output)
    oas_loss = loss_mse(tf.zeros_like(oas_output), oas_output)
    total_loss = hcp_loss + oas_loss
    return total_loss

# loss for autoencoderd
def ae_loss(h_real_data, h_regenerated_output, o_real_data, o_regenerated_output):
    # ensure the identical dtype
    h_real_data = tf.cast(h_real_data, tf.float64)
    o_real_data = tf.cast(o_real_data, tf.float64)
    h_regenerated_output = tf.cast(h_regenerated_output, tf.float64)
    o_regenerated_output = tf.cast(o_regenerated_output, tf.float64)

    # merge images
    real_data = tf.concat((h_real_data, o_real_data), axis=0)
    regenerated_output = tf.concat((h_regenerated_output, o_regenerated_output), axis=0)
    loss = loss_mse(real_data, regenerated_output)
    return loss

# Optimizers
a_optimizer = tf.keras.optimizers.Adam(1e-4)
d_optimizer = tf.keras.optimizers.Adam(1e-4)

# Train
@tf.function
def train(hcp_train_data, oas_train_data, epo, num_regions, l):
    # prepare models and list to record loss history
    ae_model = autoencoder(num_regions)
    disc_model = discriminator(num_regions)
    hcp_sub_idx = list(hcp_train_data.keys())
    oas_sub_idx = list(oas_train_data.keys())
    a_loss_ls = []
    d_loss_ls = []
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
                with tf.GradientTape() as ae_tape, tf.GradientTape() as disc_tape:
                    # reconstruct the image
                    h_recon = ae_model(h_mtx, training=True) # 900, regions
                    h_recon_tr = h_recon[:149, :] # 149, regions
                    o_recon = ae_model(o_mtx, training=True) # 149, regions
                    if h_recon_tr.shape != o_recon.shape:
                        print("Wrong shape!")

                    # discriminator outputs
                    h_recon_img = tf.expand_dims(h_recon_tr, axis=0)
                    o_recon_img = tf.expand_dims(o_recon, axis=0)
                    h_class = disc_model(h_recon_img, training=True)
                    o_class = disc_model(o_recon_img, training=True)

                    # calculate losses
                    ae_total_loss = (1 - l) * ae_loss(h_mtx, h_recon, o_mtx, o_recon)
                    disc_loss = l * di_loss(h_class, o_class)
                    # record the loss
                    a_loss_ls.append(ae_total_loss.numpy())
                    d_loss_ls.append(disc_loss.numpy())
                gradients_of_ae   = ae_tape.gradient(ae_total_loss, ae_model.trainable_variables)
                a_optimizer.apply_gradients(zip(gradients_of_ae, ae_model.trainable_variables))

                gradients_of_disc = disc_tape.gradient(disc_loss, disc_model.trainable_variables)
                d_optimizer.apply_gradients(zip(gradients_of_disc, disc_model.trainable_variables))
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
                with tf.GradientTape() as ae_tape, tf.GradientTape() as disc_tape:
                    # reconstruct the image
                    h_recon = ae_model(h_mtx, training=True) # 900, regions
                    h_recon_tr = h_recon[:149, :] # 149, regions
                    o_recon = ae_model(o_mtx, training=True) # 149, regions
                    if h_recon_tr.shape != o_recon.shape:
                        print("Wrong shape!")

                    # discriminator outputs
                    h_recon_img = tf.expand_dims(h_recon_tr, axis=0)
                    o_recon_img = tf.expand_dims(o_recon, axis=0)
                    h_class = disc_model(h_recon_img, training=True)
                    o_class = disc_model(o_recon_img, training=True)

                    # calculate losses
                    ae_total_loss = (1 - l) * ae_loss(h_mtx, h_recon, o_mtx, o_recon)
                    disc_loss = l * di_loss(h_class, o_class)
                    # record the loss
                    a_loss_ls.append(ae_total_loss.numpy())
                    d_loss_ls.append(disc_loss.numpy())
                gradients_of_ae   = ae_tape.gradient(ae_total_loss, ae_model.trainable_variables)
                a_optimizer.apply_gradients(zip(gradients_of_ae, ae_model.trainable_variables))

                gradients_of_disc = disc_tape.gradient(disc_loss, disc_model.trainable_variables)
                d_optimizer.apply_gradients(zip(gradients_of_disc, disc_model.trainable_variables))
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
                with tf.GradientTape() as ae_tape, tf.GradientTape() as disc_tape:
                    # reconstruct the image
                    h_recon = ae_model(h_mtx, training=True) # 900, regions
                    h_recon_tr = h_recon[:149, :] # 149, regions
                    o_recon = ae_model(o_mtx, training=True) # 149, regions
                    if h_recon_tr.shape != o_recon.shape:
                        print("Wrong shape!")

                    # discriminator outputs
                    h_recon_img = tf.expand_dims(h_recon_tr, axis=0)
                    o_recon_img = tf.expand_dims(o_recon, axis=0)
                    h_class = disc_model(h_recon_img, training=True)
                    o_class = disc_model(o_recon_img, training=True)

                    # calculate losses
                    ae_total_loss = (1 - l) * ae_loss(h_mtx, h_recon, o_mtx, o_recon)
                    disc_loss = l * di_loss(h_class, o_class)
                    # record the loss
                    a_loss_ls.append(ae_total_loss.numpy())
                    d_loss_ls.append(disc_loss.numpy())
                gradients_of_ae   = ae_tape.gradient(ae_total_loss, ae_model.trainable_variables)
                a_optimizer.apply_gradients(zip(gradients_of_ae, ae_model.trainable_variables))

                gradients_of_disc = disc_tape.gradient(disc_loss, disc_model.trainable_variables)
                d_optimizer.apply_gradients(zip(gradients_of_disc, disc_model.trainable_variables))
                progress_bar.update(1)
        progress_bar.close()

    print("------Training Finished------")
    # saving whole model
    ae_model.save('/data_qnap/yifeis/NAS/gan/with_param/models/autoencoder_'+str(round(l,2))+'_'+str(num_regions)+'.hdf5')
    disc_model.save('/data_qnap/yifeis/NAS/gan/with_param/models/discriminator_'+str(round(l,2))+'_'+str(num_regions)+'.hdf5')
    print("------Saved Models------")
    # save loss
    a_loss_np = np.array(a_loss_ls)
    d_loss_np = np.array(d_loss_ls)
    a_loss_df = pd.DataFrame(a_loss_np)
    d_loss_df = pd.DataFrame(d_loss_np)
    a_loss_df.to_csv('/data_qnap/yifeis/NAS/gan/with_param/losses/autoencoder_loss_'+str(round(l,2))+'_'+str(num_regions)+'.csv', index=False)
    d_loss_df.to_csv('/data_qnap/yifeis/NAS/gan/with_param/losses/discriminator_loss_'+str(round(l,2))+'_'+str(num_regions)+'.csv', index=False)
    print("------Saved Loss------")
    return (autoencoder, discriminator, a_loss_np, d_loss_np)


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
oas_train_data_360, oas_train_sessions_360 = load_OAS_data("HC", oas_train_subjects[50:100], "2mm", "tf")
oas_train_data_379, oas_train_sessions_379 = load_OAS_data("HC", oas_train_subjects[50:100], 'surf', "tf")

# train
lambda_value = 0
for n in range(99):
    lambda_value += 0.01
    lambda_value = round(lambda_value, 2)
    epochs = 20
    autoencoder_360, discriminator_360, ae_loss_360, disc_loss_360 = train(hcp_train_data_360, oas_train_data_360, epochs, 360, lambda_value)
    autoencoder_379, discriminator_379, ae_loss_379, disc_loss_379 = train(hcp_train_data_379, oas_train_data_379, epochs, 379, lambda_value)

    #    Plot the history to verify they converged
    plot1 = plt.figure(1)
    plt.plot(ae_loss_360, label="360 Regions")
    plt.plot(ae_loss_379, label="379 Regions")

    plt.xlabel("# of iterations")
    plt.ylabel("MSE")
    plt.title('Loss of Autoencoder, Lambda = {}'.format(lambda_value))
    plt.legend()
    plt.savefig('/data_qnap/yifeis/NAS/gan/with_param/losses/plot_autoencoder_loss_'+str(lambda_value)+'.png')
    plt.clf()
    plot2 = plt.figure(2)
    plt.plot(disc_loss_360, label="360 Regions")
    plt.plot(disc_loss_379, label="379 Regions")
    plt.xlabel("# of iterations")
    plt.ylabel("MSE")
    plt.title('Loss of Discriminator, Lambda = {}'.format(lambda_value))
    plt.legend()
    plt.savefig('/data_qnap/yifeis/NAS/gan/with_param/losses/plot_discriminator_loss_'+str(lambda_value)+'.png')
    plt.clf()
