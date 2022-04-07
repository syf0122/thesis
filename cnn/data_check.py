import os
import sys
import glob
import nibabel as nib
from sphericalunet.utils.vtk import read_vtk, write_vtk, resample_label
import numpy as np

# # read gifti
# gdata = nib.load('/data_qnap/yifeis/HCP_7T/100610/rest1_left.func.gii')
# coords = np.array(gdata.agg_data())
# print(coords.shape)
# count = 0
# for i in range(coords.shape[1]):
# 	n = coords[:, i]
# 	if np.all(n == 0):
# 		print(i)
# 		count += 1
# print(f'there are {count} vertices with 0 values.')
# quit()

# def surf_data_from_cifti(data, axis, surf_name):
#     assert isinstance(axis, nib.cifti2.BrainModelAxis)
#     for name, data_indices, model in axis.iter_structures():
#         if name == surf_name:
#             data = data.T[data_indices] # get the corresponding slice from the whole matrix
#             vtx_indices = model.vertex
#             # print((vtx_indices.max() + 1,)) # number of rows
#             # print(data.shape[1:]) # number of columns
#             surf_data = np.zeros((vtx_indices.max() + 1,) + data.shape[1:], dtype = data.dtype)
#             surf_data[vtx_indices] = data
#             return surf_data # numpy arrary
#     raise ValueError(f"No structure named {surf_name}")
#
# def decompose_cifti(img):
#     data = img.get_fdata(dtype=np.float32) # extract fMRI timeseries to numpy arrary
#     brain_models = img.header.get_axis(1)
#     return (surf_data_from_cifti(data, brain_models, "CIFTI_STRUCTURE_CORTEX_LEFT"),
#             surf_data_from_cifti(data, brain_models, "CIFTI_STRUCTURE_CORTEX_RIGHT"))
#
# f_data = nib.load("/data_qnap/yifeis/HCP_7T/100610/rest1_164k.dtseries.nii")
# left, right = decompose_cifti(f_data)
# print(left.shape)
# count = 0
# for i in range(right.shape[0]):
# 	n = right[i]
# 	if np.all(n == 0):
# 		print(i)
# 		count += 1
# print(f'there are {count} vertices with 0 values.')
# quit()


# f_data = read_vtk('/data_qnap/yifeis/HCP_7T/100610/rest1_left.vtk')['cdata']
# f_data_norm = (f_data - np.mean(f_data,axis=0))/np.std(f_data,axis=0)
# print(f_data.shape)
# print(np.max(f_data))
# print(np.min(f_data))
# first = f_data_norm[:, 0]
# print(first.shape)
# print(np.mean(first))
# print(np.std(first))
# print(np.max(f_data_norm))
# print(np.min(f_data_norm))
# quit()

dir = '/data_qnap/yifeis/NAS/HCP_7T/'
subjects = os.listdir(dir)
subjects.sort()
print(f'There are {len(subjects)}.')
type = sys.argv[1]
print(dir)
vtk_dir = {}
for sub in subjects:
	sub_dir = dir + sub
	all_sub_files = os.listdir(sub_dir)
	sub_vtk_dir = []
	for file in all_sub_files:
	    if file.endswith(".vtk") and type in file:
	        sub_vtk_dir.append(os.path.join(sub_dir, file))
	vtk_dir[sub] = sub_vtk_dir

for n in vtk_dir:
	print(n)
	print(len(vtk_dir[n]))


# e1_data = read_vtk("/data_qnap/yifeis/spherical_cnn/example_data/test1.rh.160k.vtk")
# e2_data = read_vtk("/data_qnap/yifeis/spherical_cnn/example_data/test2.lh.160k.vtk")
# e3_data = read_vtk("/data_qnap/yifeis/spherical_cnn/example_data/test3.lh.160k.vtk")
# f1_data = read_vtk('/data_qnap/yifeis/HCP_7T/100610/rest1_left.vtk')
# f2_data = read_vtk('/data_qnap/yifeis/HCP_7T/102816/rest1_left.vtk')
# f3_data = read_vtk('/data_qnap/yifeis/HCP_7T/169747/rest1_left.vtk')
# #
# print(e1_data['faces'][:5])
# print(e2_data['faces'][:5])
# print(e3_data['faces'][:5])
# print()
# print(f1_data['faces'][:5])
# print(f2_data['faces'][:5])
# print(f3_data['faces'][:5])
# print()

# print(f1_data['cdata'][:2, :5])
# print(f2_data['cdata'][:2, :5])
#
# print(np.max(f1_data['cdata']))
# print(np.max(f2_data['cdata']))
# print(np.min(f1_data['cdata']))
# print(np.min(f2_data['cdata']))

# print(np.max(e1_data['vertices']))
# print(np.max(e2_data['vertices']))
# print(np.max(e3_data['vertices']))
# print(np.max(f_data['vertices']))
# print()
# print(np.min(e1_data['vertices']))
# print(np.min(e2_data['vertices']))
# print(np.min(e3_data['vertices']))
# print(np.min(f_data['vertices']))
# print(e1_data.keys())
# print(e2_data.keys())
# print(f_data.keys())
# print()
#
# print(e1_data['vertices'].shape)
# print(e2_data['vertices'].shape)
# print(f_data['vertices'].shape)
# print()
#
# print(e1_data['faces'].shape)
# print(e2_data['faces'].shape)
# print(f_data['faces'].shape)
# print()
#
# print(e1_data['Normals'].shape)
# print(e2_data['Normals'].shape)
# print(f_data['normals'].shape)
