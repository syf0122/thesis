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
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import tensorflow_docs.modeling
import tensorflow_docs.plots
from tensorflow.keras.models import load_model

# check
autoencoder = load_model('/data_qnap/yifeis/ae_relu_trained_with_50/model/autoencoder.hdf5', compile=False)
encoder = load_model('/data_qnap/yifeis/ae_relu_trained_with_50/model/encoder.hdf5', compile=False)
decoder = load_model('/data_qnap/yifeis/ae_relu_trained_with_50/model/decoder.hdf5', compile=False)
decoder_80 = load_model('/data_qnap/yifeis/ae_relu_trained_with_50/model/decoder_80.hdf5', compile=False)
decoder_160 = load_model('/data_qnap/yifeis/ae_relu_trained_with_50/model/decoder_160.hdf5', compile=False)

print(encoder.summary())
print(decoder.summary())
print(decoder_80.summary())
print(decoder_160.summary())

print("Check Config")
for l1, l2 in zip(encoder.layers, autoencoder.layers[:4]):
    print(l1.get_config() == l2.get_config())

for l1, l2 in zip(decoder.layers[1:], autoencoder.layers[4:]):
    print(l1.get_config() == l2.get_config())

for l1, l2 in zip(decoder_80.layers, autoencoder.layers[:5]):
    print(l1.get_config() == l2.get_config())

for l1, l2 in zip(decoder_160.layers, autoencoder.layers[:6]):
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

check_equl(decoder_80.layers[1].get_weights(), autoencoder.layers[1].get_weights())
check_equl(decoder_80.layers[2].get_weights(), autoencoder.layers[2].get_weights())
check_equl(decoder_80.layers[3].get_weights(), autoencoder.layers[3].get_weights())
check_equl(decoder_80.layers[4].get_weights(), autoencoder.layers[4].get_weights())

check_equl(decoder_160.layers[1].get_weights(), autoencoder.layers[1].get_weights())
check_equl(decoder_160.layers[2].get_weights(), autoencoder.layers[2].get_weights())
check_equl(decoder_160.layers[3].get_weights(), autoencoder.layers[3].get_weights())
check_equl(decoder_160.layers[4].get_weights(), autoencoder.layers[4].get_weights())
check_equl(decoder_160.layers[5].get_weights(), autoencoder.layers[5].get_weights())
