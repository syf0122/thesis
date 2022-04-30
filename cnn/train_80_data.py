import scipy.io as sio
import numpy as np
import os
from tqdm import tqdm
import nibabel as nib
import hcp_utils as hcp

def helper_gifti_load(file_loc):
    # read gifti
    f_data = nib.load(file_loc)
    f_data = np.array(f_data.agg_data())
    f_data = hcp.normalize(f_data)
    f_data = f_data.T
    f_data = np.nan_to_num(f_data) # convert to zero for those vertices with all 0 values
    return f_data

# hemisphere
hemi = 'right'

# sub
dir = '/home/yifeis/DGX/NAS/HCP_7T/'
subjects = os.listdir(dir)
subjects.sort()

dir_2 = '/home/yifeis/DGX/HCP_7T/'
subjects_2 = os.listdir(dir_2)
subjects_2.sort()
subjects_2 = subjects_2[:30]
print(f'There are {len(subjects)+len(subjects_2)} HCP_7T preprocessed subjects.')

# gifti
g_dir = {}
for sub in subjects:
    sub_dir = dir + sub
    all_sub_files = os.listdir(sub_dir)
    all_sub_files.sort()
    sub_gif_dir = []
    for file in all_sub_files:
        if file.endswith(".func.gii") and 'rest1' in file and hemi in file:
            sub_gif_dir.append(os.path.join(sub_dir, file))
    g_dir[sub] = sub_gif_dir

for sub in subjects_2:
    sub_dir = dir_2 + sub
    all_sub_files = os.listdir(sub_dir)
    all_sub_files.sort()
    sub_gif_dir = []
    for file in all_sub_files:
        if file.endswith(".func.gii") and 'rest1' in file and hemi in file:
            sub_gif_dir.append(os.path.join(sub_dir, file))
    g_dir[sub] = sub_gif_dir

rest_sub = []
for sub in g_dir:
    print(g_dir[sub])
    if len(g_dir[sub]) == 1:
        rest_sub.append(sub)
print(f'There are currently {len(rest_sub)} subjects with rest session converted to gifti.')

'''
    prepare the train data
    first 50 subjects
    4 rs-fMRI sessions
'''
train_data = []
progress_bar = tqdm(range(80))
for sub in rest_sub:
    rs_files = g_dir[sub]
    for r in rs_files:
        train_data.append(helper_gifti_load(r))
        progress_bar.update(1)
progress_bar.close()
# concatenate all data matrices
progress_bar = tqdm(range(len(train_data)))
all_data = train_data[0]
progress_bar.update(1)
for d in train_data[1:]:
    all_data = np.concatenate((all_data, d), axis=1)
    progress_bar.update(1)
progress_bar.close()

print(all_data.shape)
np.save('/home/yifeis/DGX/spherical_cnn/test/right_80_train_gifti_data_r1.npy', all_data)
