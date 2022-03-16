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


# loading whole model
print("Load Trained Model")
autoencoder = load_model('/data_qnap/yifeis/ae_model/autoencoder_model_1.hdf5', compile=False)
encoder = load_model('/data_qnap/yifeis/ae_model/encoder_model_1.hdf5', compile=False)
decoder = load_model('/data_qnap/yifeis/ae_model/decoder_model_1.hdf5', compile=False)
print("Loading Successful!")

print("Check Config")
for l1, l2 in zip(encoder.layers, autoencoder.layers[:4]):
    print(l1.get_config() == l2.get_config())

print()
print("Check Weights")
def check_equl (w_1, w_2):
    if len(w_1) != len(w_2):
        print("different layers")
        return
    # check the first element
    array_1 = list(w_1[0])
    array_2 = list(w_2[0])
    if len(array_1) != len(array_2):
        print(False)
        return
    for i in range(len(array_1)):
        a_1 = list(array_1[i])
        a_2 = list(array_2[i])
        if a_1 != a_2:
            print(False)
            return

    #check the second element
    array_1 = list(w_1[1])
    array_2 = list(w_2[1])
    if array_1 != array_2:
        print(False)
        return
    print(True)

check_equl(encoder.layers[1].get_weights(), autoencoder.layers[1].get_weights())
check_equl(encoder.layers[2].get_weights(), autoencoder.layers[2].get_weights())
check_equl(encoder.layers[3].get_weights(), autoencoder.layers[3].get_weights())

check_equl(decoder.layers[1].get_weights(), autoencoder.layers[4].get_weights())
check_equl(decoder.layers[2].get_weights(), autoencoder.layers[5].get_weights())
check_equl(decoder.layers[3].get_weights(), autoencoder.layers[6].get_weights())


mtx1_p = np.load('/data_qnap/yifeis/new/processed/102816/rest3_p.npy')
mtx1_p = (mtx1_p - mtx1_p.min()) / (mtx1_p.max() - mtx1_p.min())

# latent_vector
latent_vector = encoder.predict(mtx1_p)
# decoder output
reconstructed_data = autoencoder.predict(mtx1_p)
print()
print(latent_vector)
print()
print(reconstructed_data)

reconstructed_data = decoder.predict(latent_vector)
print()
print(reconstructed_data)
