import scipy.io as sio
import numpy as np
import os
from tqdm import tqdm
import nibabel as nib
from sphericalunet.utils.vtk import read_vtk, write_vtk, resample_label

def helper_load(file_loc):
    # read vtk
    f_data = read_vtk(file_loc)['cdata']
    # normalization
    f_data = (f_data - np.mean(f_data,axis=0))/np.std(f_data,axis=0)
    # print(f_data.shape) # (163842, 900)
    return f_data

def helper_gifti_load(file_loc):
    # read gifti
    f_data = nib.load(file_loc)
    f_data = np.array(f_data.agg_data()).T
    # normalization
    f_data = (f_data - np.mean(f_data,axis=0))/np.std(f_data,axis=0)
    # print(f_data.shape) # (163842, 900)
    return f_data


# # test
# v_data = helper_load("/data_qnap/yifeis/NAS/HCP_7T/100610/rest1_left.vtk")
# print()
# g_data = helper_gifti_load("/data_qnap/yifeis/NAS/HCP_7T/100610/rest1_left.func.gii")
# print()
# print(v_data[0].shape)
# print(v_data[0].mean())
# print(v_data[:, 0].shape)
# print(v_data[:, 0].mean())
# print()
# print(g_data[0].shape)
# print(g_data[0].mean())
# print(g_data[:, 0].shape)
# print(g_data[:, 0].mean())
# quit()

# hemisphere
hemi = 'right'

# sub
dir = '/data_qnap/yifeis/NAS/HCP_7T/'
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
np.save('/data_qnap/yifeis/spherical_cnn/test/first_'+str(num_of_sub)+'_'+hemi+'_train_gifti_data.npy', all_data)

# for fold in range(5):
#     # get the training subjects for each fold
#     test_subs  = subjects[fold*10: fold*10 + 10]
#       print(test_subs)
#     train_subs = []
#     for s in subjects:
#         if s not in test_subs:
#             train_subs.append(s)
#
#     # load all rs data for left and right hemisphere
#     train_data_l = []
#     train_data_r = []
#     progress_bar = tqdm(range(len(train_subs) * 8))
#     for s in train_subs:
#         # left
#         train_data_l.append(helper_load(dir+s+'/rest1_left.vtk'))
#         progress_bar.update(1)
#         train_data_l.append(helper_load(dir+s+'/rest2_left.vtk'))
#         progress_bar.update(1)
#         train_data_l.append(helper_load(dir+s+'/rest3_left.vtk'))
#         progress_bar.update(1)
#         train_data_l.append(helper_load(dir+s+'/rest4_left.vtk'))
#         progress_bar.update(1)
#
#         # right
#         train_data_r.append(helper_load(dir+s+'/rest1_right.vtk'))
#         progress_bar.update(1)
#         train_data_r.append(helper_load(dir+s+'/rest2_right.vtk'))
#         progress_bar.update(1)
#         train_data_r.append(helper_load(dir+s+'/rest3_right.vtk'))
#         progress_bar.update(1)
#         train_data_r.append(helper_load(dir+s+'/rest4_right.vtk'))
#         progress_bar.update(1)
#     progress_bar.close()
#
#     # concatenate all data matrices
#     # left
#     progress_bar = tqdm(range(len(train_data_l)))
#     all_data_l = train_data_l[0]
#     progress_bar.update(1)
#     for d in train_data_l[1:]:
#         all_data_l = np.concatenate((all_data_l, d), axis=1)
#         progress_bar.update(1)
#     progress_bar.close()
#
#     # right
#     progress_bar = tqdm(range(len(train_data_r)))
#     all_data_r = train_data_r[0]
#     progress_bar.update(1)
#     for d in train_data_r[1:]:
#         all_data_r = np.concatenate((all_data_r, d), axis=1)
#         progress_bar.update(1)
#     progress_bar.close()
#
#
#     # save as .npy file
#     np.save('/data_qnap/yifeis/spherical_cnn/train_data/train_'+str(fold+1)+'_l.npy', all_data_l)
#     np.save('/data_qnap/yifeis/spherical_cnn/train_data/train_'+str(fold+1)+'_r.npy', all_data_r)
