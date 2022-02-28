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
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

'''
    Load Training Data
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

'''
    Training the Autoencoder model
    For both 360 and 379 regions
'''
def train_360(train_data, epo, name):
    train_sub = []
    for n in train_data:
        print(n)
        train_sub.append(train_data[n])
    print("there are " + str(len(train_sub)) + " subjects for training.")
    # encoder
    input_data = Input(shape=(360,))
    encoder1 = Dense(180, activation='sigmoid')(input_data)
    encoder2 = Dense(90, activation='sigmoid')(encoder1)
    encoder3 = Dense(45, activation='sigmoid')(encoder2)

    # decoder
    decoder1 = Dense(90, activation='sigmoid')(encoder3)
    decoder2 = Dense(180, activation='sigmoid')(decoder1)
    decoder3 = Dense(360, activation='sigmoid')(decoder2)

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
    encoded_input = Input(shape=(45,))
    decoder_layer1 = autoencoder.layers[-3]
    decoder_layer2 = autoencoder.layers[-2]
    decoder_layer3 = autoencoder.layers[-1]
    decoder = Model(inputs=encoded_input, outputs=decoder_layer3(decoder_layer2(decoder_layer1(encoded_input))))


    # saving whole model
    autoencoder.save('/data_qnap/yifeis/NAS/ae/train_hc/autoencoder_360.hdf5')
    encoder.save('/data_qnap/yifeis/NAS/ae/train_hc/encoder_360.hdf5')
    decoder.save('/data_qnap/yifeis/NAS/ae/train_hc/decoder_360.hdf5')
    print("Saving Models for 360 Regions Successful!")

    ## total history
    # create an empty dict to save all three history dicts into
    total_history_dict = dict()
    for some_key in hist_ls[0].history.keys():
        current_values = [] # to save values from all three hist dicts
        for hist_dict in hist_ls:
            hist_dict = hist_dict.history
            current_values += hist_dict[some_key]
        total_history_dict[some_key] = current_values
    return (autoencoder, encoder, decoder, total_history_dict)

def train_379(train_data, epo, name):
    train_sub = [] # each element is the list of data for each subject
    for n in train_data:
        print(n)
        train_sub.append(train_data[n])
    print("there are " + str(len(train_sub)) + " subjects for training.")
    # encoder
    input_data = Input(shape=(379,))
    encoder1 = Dense(160, activation='sigmoid')(input_data)
    encoder2 = Dense(80, activation='sigmoid')(encoder1)
    encoder3 = Dense(40, activation='sigmoid')(encoder2)

    # decoder
    decoder1 = Dense(80, activation='sigmoid')(encoder3)
    decoder2 = Dense(160, activation='sigmoid')(decoder1)
    decoder3 = Dense(379, activation='sigmoid')(decoder2)

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


    # saving whole model
    autoencoder.save('/data_qnap/yifeis/NAS/ae/train_hc/autoencoder_379.hdf5')
    encoder.save('/data_qnap/yifeis/NAS/ae/train_hc/encoder_379.hdf5')
    decoder.save('/data_qnap/yifeis/NAS/ae/train_hc/decoder_379.hdf5')
    print("Saving Models for 379 Regions Successful!")

    ## total history
    # create an empty dict to save all three history dicts into
    total_history_dict = dict()
    for some_key in hist_ls[0].history.keys():
        current_values = [] # to save values from all three hist dicts
        for hist_dict in hist_ls:
            hist_dict = hist_dict.history
            current_values += hist_dict[some_key]
        total_history_dict[some_key] = current_values
    return (autoencoder, encoder, decoder, total_history_dict)

def train_old_360(autoencoder, train_data, epo, name): # need update
    train_sub = []
    for n in train_data:
        print(n)
        train_sub.append(train_data[n])
    print("there are " + str(len(train_sub)) + " subjects for training.")

    # train
    hist_ls = []
    autoencoder.compile(optimizer='adam', loss='mean_squared_error',metrics=[losses.MeanSquaredError(), 'accuracy'])
    sub_idx = list(range(len(train_sub)))
    # round 1
    for sub_i in sub_idx: # for each subject in origional order
        sub = train_sub[sub_i]
        print("First round: training with subject " + str(train_subjects[sub_i]))
        for tr in sub: # train the model with each matrix from this subject for epo epochs
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
    # saving whole model
    autoencoder.save('/data_qnap/yifeis/NAS/ae/train_50_hc/autoencoder_360.hdf5')
    print("Saving Models for 360 Regions Successful!")

    ## total history
    # create an empty dict to save all three history dicts into
    total_history_dict = dict()
    for some_key in hist_ls[0].history.keys():
        current_values = [] # to save values from all three hist dicts
        for hist_dict in hist_ls:
            hist_dict = hist_dict.history
            current_values += hist_dict[some_key]
        total_history_dict[some_key] = current_values
    return (autoencoder, total_history_dict)

def train_old_379(autoencoder, train_data, epo, name): # need update
    train_sub = []
    for n in train_data:
        print(n)
        train_sub.append(train_data[n])
    print("there are " + str(len(train_sub)) + " subjects for training.")

    # train
    hist_ls = []
    autoencoder.compile(optimizer='adam', loss='mean_squared_error',metrics=[losses.MeanSquaredError(), 'accuracy'])
    sub_idx = list(range(len(train_sub)))
    # round 1
    for sub_i in sub_idx: # for each subject in origional order
        sub = train_sub[sub_i]
        print("First round: training with subject " + str(train_subjects[sub_i]))
        for tr in sub: # train the model with each matrix from this subject for epo epochs
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
    # saving whole model
    autoencoder.save('/data_qnap/yifeis/NAS/ae/train_50_hc/autoencoder_379.hdf5')
    print("Saving Models for 379 Regions Successful!")

    ## total history
    # create an empty dict to save all three history dicts into
    total_history_dict = dict()
    for some_key in hist_ls[0].history.keys():
        current_values = [] # to save values from all three hist dicts
        for hist_dict in hist_ls:
            hist_dict = hist_dict.history
            current_values += hist_dict[some_key]
        total_history_dict[some_key] = current_values
    return (autoencoder, total_history_dict)

'''
    Load Trained Models (previously trained using 50 HCP subjects rs-fMRI)
'''
autoencoder_360_prev = load_model('/data_qnap/yifeis/NAS/ae/train_50/autoencoder_360.hdf5', compile=False)
encoder_360_prev = load_model('/data_qnap/yifeis/NAS/ae/train_50/encoder_360.hdf5', compile=False)
decoder_360_prev = load_model('/data_qnap/yifeis/NAS/ae/train_50/decoder_360.hdf5', compile=False)

autoencoder_379_prev = load_model('/data_qnap/yifeis/NAS/ae/train_50/autoencoder_379.hdf5', compile=False)
encoder_379_prev = load_model('/data_qnap/yifeis/NAS/ae/train_50/encoder_379.hdf5', compile=False)
decoder_379_prev = load_model('/data_qnap/yifeis/NAS/ae/train_50/decoder_379.hdf5', compile=False)
print("Loading Successful!")
# print(autoencoder_360.summary())
# print(autoencoder_379.summary())

'''
    Load Data
'''
# Train OAS subjects
train_subjects = os.listdir("/data_qnap/yifeis/NAS/data/HC/")
train_subjects.sort()
# print(train_subjects[50:100])
# print()

 # load data
train_data_360, train_sessions_360 = load_OAS_data("HC", train_subjects[50:100], "2mm", "tf")
train_data_379, train_sessions_379 = load_OAS_data("HC", train_subjects[50:100], 'surf', "tf")

# for n in train_subjects[50:100]:
#     print(n)
#     print(len(train_data_360[n]) == 2)
#     print(len(train_data_379[n]) == 2)
#     print(len(train_sessions_360[n]) == 2)
#     print(len(train_sessions_379[n]) == 2)
#     print(train_data_360[n][0].shape == (149, 360))
#     print(train_data_379[n][0].shape == (149, 379))
#     print(train_data_360[n][1].shape == (149, 360))
#     print(train_data_379[n][1].shape == (149, 379))
#     print(train_sessions_360[n])
#     print(train_sessions_379[n])
#     print()
# quit()

'''
    Train the HC only model
    3 rounds
    50 subjects randomized order
    Each matrix train for 40 epochs in each round
'''
epochs = 175
autoencoder_360, encoder_360, decoder_360, total_history_360  = train_360(train_data_360, epochs, 'basic_ae')
autoencoder_379, encoder_379, decoder_379, total_history_379  = train_379(train_data_379, epochs, 'basic_ae')

'''
    Plot the history to verify they converged
'''
# print(total_history.keys())
plot1 = plt.figure(1)
plt.plot(total_history_360["mean_squared_error"])
plt.xlabel("# of iterations")
plt.ylabel("MSE")
plt.title('Autoencoder Trained with 360 Regions')
plt.savefig('/data_qnap/yifeis/NAS/ae/train_hc/train_360_MSE.png')
plt.clf()

plot2 = plt.figure(2)
plt.plot(total_history_379["mean_squared_error"])
plt.xlabel("# of iterations")
plt.ylabel("MSE")
plt.title('Autoencoder Trained with 379 Regions')
plt.savefig('/data_qnap/yifeis/NAS/ae/train_hc/train_379_MSE.png')
plt.clf()



# '''
#     Train the Previous train_50 with HC data
#     3 rounds
#     50 subjects randomized order
#     Each matrix train for 40 epochs in each round
# '''
# epochs = 100
# autoencoder_360, total_history_360  = train_old_360(autoencoder_360_prev, train_data_360, epochs, 'basic_ae')
# autoencoder_379, total_history_379  = train_old_379(autoencoder_379_prev, train_data_379, epochs, 'basic_ae')
#
# '''
#     Plot the history to verify they converged
# '''
# # print(total_history.keys())
# plot1 = plt.figure(1)
# plt.plot(total_history_360["mean_squared_error"])
# plt.xlabel("# of iterations")
# plt.ylabel("MSE")
# plt.title('Autoencoder Trained with 360 Regions')
# plt.savefig('/data_qnap/yifeis/NAS/ae/train_50_hc/train_360_MSE.png')
# plt.clf()
#
# plot2 = plt.figure(2)
# plt.plot(total_history_379["mean_squared_error"])
# plt.xlabel("# of iterations")
# plt.ylabel("MSE")
# plt.title('Autoencoder Trained with 379 Regions')
# plt.savefig('/data_qnap/yifeis/NAS/ae/train_50_hc/train_379_MSE.png')
#
# # extract and save the encoder and decoder
# autoencoder_360 = load_model('/data_qnap/yifeis/NAS/ae/train_50_hc/autoencoder_360.hdf5', compile=False)
# autoencoder_379 = load_model('/data_qnap/yifeis/NAS/ae/train_50_hc/autoencoder_379.hdf5', compile=False)
#
# print("Loading Successful!")
#
# def extraction (autoencoder, s, l):
#     # get encoder model
#     test_input = Input(shape=(s,))
#     encoder_layer1 = autoencoder.layers[1]
#     encoder_layer2 = autoencoder.layers[2]
#     encoder_layer3 = autoencoder.layers[3]
#     encoder = Model(inputs=test_input, outputs=encoder_layer3(encoder_layer2(encoder_layer1(test_input))))
#
#     # get decoder model
#     encoded_input = Input(shape=(l,))
#     decoder_layer1 = autoencoder.layers[-3]
#     decoder_layer2 = autoencoder.layers[-2]
#     decoder_layer3 = autoencoder.layers[-1]
#     decoder = Model(inputs=encoded_input, outputs=decoder_layer3(decoder_layer2(decoder_layer1(encoded_input))))
#
#     encoder.save('/data_qnap/yifeis/NAS/ae/train_50_hc/encoder_'+str(s)+'.hdf5')
#     decoder.save('/data_qnap/yifeis/NAS/ae/train_50_hc/decoder_'+str(s)+'.hdf5')
#
# extraction(autoencoder_360, 360, 45)
# extraction(autoencoder_379, 379, 40)
