import os
import numpy as np

# get all se results
se_files = os.listdir('/data_qnap/yifeis/spherical_cnn/results/SE/')
se_files.sort()
for f in se_files:
    if f.endswith('.npy'):
        # data = np.load(f)
        # data = data.mean(axis=0)
        # print(data.shape)
        print('/data_qnap/yifeis/spherical_cnn/results/SE/avg/' + f[:-4] + "_avg.npy")
        # data.save('/data_qnap/yifeis/spherical_cnn/results/SE/avg/' + f[:-4] + "_avg.npy")
