import scipy.io as sio
import numpy as np
import os
from tqdm import tqdm
import nibabel as nib
from sphericalunet.utils.vtk import read_vtk, write_vtk, resample_label
import hcp_utils as hcp

def helper_load(file_loc):
    # read vtk
    f_data = read_vtk(file_loc)['cdata']
    # # normalization
    f_data = f_data.T
    f_data = hcp.normalization(f_data)
    f_data = f_data.T
    return f_data

def helper_gifti_load(file_loc):
    # read gifti
    f_data = nib.load(file_loc)
    f_data = np.array(f_data.agg_data())
    f_data = hcp.normalize(f_data)
    f_data = f_data.T
    f_data = np.nan_to_num(f_data) # convert to zero for those vertices with all 0 values
    f_data = (f_data - np. min(f_data)) / (np. max(f_data) - np. min(f_data)) # normalize to 0-1
    return f_data

# hemisphere
hemi = 'right'

# sub
dir = '/home/yifeis/DGX/NAS/HCP_7T/'
subjects = os.listdir(dir)
subjects.sort()
print(f'There are {len(subjects)} HCP_7T preprocessed subjects.')

# vtk_dir = {}
# for sub in subjects:
# 	sub_dir = dir + sub
# 	all_sub_files = os.listdir(sub_dir)
# 	sub_vtk_dir = []
# 	for file in all_sub_files:
# 	    if file.endswith(".vtk") and 'rest' in file and hemi in file:
# 	        sub_vtk_dir.append(os.path.join(sub_dir, file))
# 	vtk_dir[sub] = sub_vtk_dir
#
# rest_sub = []
# for sub in vtk_dir:
# 	if len(vtk_dir[sub]) == 4:
# 		rest_sub.append(sub)
# print(f'There are currently {len(rest_sub)} subjects with rest session converted to vtk.')
# # print(rest_sub)
#
# '''
#     prepare the train data
#     20 subjects
#     RIGHT hemisphere
#     4 rs-fMRI sessions
# '''
# # get the first 10 subjects
# num_of_sub = 20
# subs = rest_sub[:num_of_sub]
# print(subs)
# train_data = []
# progress_bar = tqdm(range(len(subs)*4))
# for sub in subs:
#     rs_files = vtk_dir[sub]
#     for r in rs_files:
#         train_data.append(helper_load(r))
#         progress_bar.update(1)
# progress_bar.close()
# # concatenate all data matrices
# progress_bar = tqdm(range(len(train_data)))
# all_data = train_data[0]
# progress_bar.update(1)
# for d in train_data[1:]:
#     all_data = np.concatenate((all_data, d), axis=1)
#     progress_bar.update(1)
# progress_bar.close()
# print(all_data.shape)
# np.save('/data_qnap/yifeis/spherical_cnn/test/first_'+str(num_of_sub)+'_'+hemi+'_train_data.npy', all_data)

# gifti
g_dir = {}
for sub in subjects:
    sub_dir = dir + sub
    all_sub_files = os.listdir(sub_dir)
    all_sub_files.sort()
    sub_vtk_dir = []
    for file in all_sub_files:
        if file.endswith(".func.gii") and 'rest' in file and hemi in file:
            sub_vtk_dir.append(os.path.join(sub_dir, file))
    g_dir[sub] = sub_vtk_dir

rest_sub = []
for sub in g_dir:
	if len(g_dir[sub]) == 4:
		rest_sub.append(sub)
print(f'There are currently {len(rest_sub)} subjects with rest session converted to gifti.')
# print(rest_sub)

'''
    prepare the train data
    20 subjects
    RIGHT hemisphere
    4 rs-fMRI sessions
'''
# get the first 10 subjects
num_of_sub = 20
subs = rest_sub[:num_of_sub]
print(subs)
train_data = []
progress_bar = tqdm(range(len(subs)*4))
for sub in subs:
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
np.save('/home/yifeis/DGX/spherical_cnn/test/first_'+str(num_of_sub)+'_'+hemi+'_train_gifti_01_data.npy', all_data)
