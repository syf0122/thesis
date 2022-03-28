import torch
import torch.nn as nn
import torchvision
import scipy.io as sio
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import Unet_160k
from sphericalunet.utils.vtk import read_vtk, write_vtk, resample_label


################################################################
""" hyper-parameters """
cuda = torch.device('cuda:0')
in_channels = 1
out_channels = 1
learning_rate = 0.001
################################################################

class BrainSphere(torch.utils.data.Dataset):

	def __init__(self, root, subjects):
		# load all rs data
		self.fmri_data = []
		progress_bar = tqdm(range(len(subjects) * 1))
		for s in subjects:
			f_data = read_vtk(root + s + '/rest1_left.vtk')['cdata']
			f_data = (f_data - np.mean(f_data,axis=0))/np.std(f_data,axis=0)
			self.fmri_data.append(f_data)
			progress_bar.update(1)
			# self.fmri_data.append(read_vtk(root + s + '/rest1_right.vtk')['cdata'])
			# progress_bar.update(1)
			# self.fmri_data.append(read_vtk(root + s + '/rest2_left.vtk')['cdata'])
			# progress_bar.update(1)
			# self.fmri_data.append(read_vtk(root + s + '/rest2_right.vtk')['cdata'])
			# progress_bar.update(1)
			# self.fmri_data.append(read_vtk(root + s + '/rest3_left.vtk')['cdata'])
			# progress_bar.update(1)
			# self.fmri_data.append(read_vtk(root + s + '/rest3_right.vtk')['cdata'])
			# progress_bar.update(1)
			# self.fmri_data.append(read_vtk(root + s + '/rest4_left.vtk')['cdata'])
			# progress_bar.update(1)
			# self.fmri_data.append(read_vtk(root + s + '/rest4_right.vtk')['cdata'])
			# progress_bar.update(1)
		progress_bar.close()

		# concatenate all data matrices
		progress_bar = tqdm(range(len(self.fmri_data)))
		self.all_data = self.fmri_data[0]
		progress_bar.update(1)
		for d in self.fmri_data[1:]:
			self.all_data = np.concatenate((self.all_data, d), axis=1)
			progress_bar.update(1)
		progress_bar.close()

	def __getitem__(self, index):
		one_tp_data = self.all_data[:, index]
		one_tp_data = torch.from_numpy(one_tp_data).unsqueeze(1)
		print(one_tp_data.shape) # (163438, 1)
		return one_tp_data, one_tp_data

	def __len__(self):
		return self.all_data.shape[1]

# sub
dir = '/data_qnap/yifeis/HCP_7T/'
subjects = os.listdir(dir)
subjects.sort()
print(f'There are {len(subjects)}.')

vtk_dir = {}
for sub in subjects:
	sub_dir = dir + sub
	all_sub_files = os.listdir(sub_dir)
	sub_vtk_dir = []
	for file in all_sub_files:
	    if file.endswith(".vtk") and 'rest' in file:
	        sub_vtk_dir.append(os.path.join(sub_dir, file))
	vtk_dir[sub] = sub_vtk_dir

rest_sub = []
for sub in vtk_dir:
	if len(vtk_dir[sub]) == 8:
		rest_sub.append(sub)
print(f'There are currently {len(rest_sub)} subjects with rest1 session converted to vtk.')

print(rest_sub)
# get the data for training and testing
train_dataset = BrainSphere(dir, rest_sub[:2]) # number of subjects * 4 * 2 * 900
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, pin_memory=True) # each sample (163438, 1)
print(f'Training dataset shape: {train_dataset.all_data.shape}')

# get the model
model = Unet_160k(in_ch=in_channels, out_ch=out_channels)
print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
model.cuda(cuda)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.2, patience=1, verbose=True, threshold=0.0001, threshold_mode='rel', min_lr=0.000001)

# training
def train_step(data, target):
	model.train()
	data, target = data.cuda(cuda), target.cuda(cuda)
	prediction = model(data)
	print('input shape')
	print(data.shape)
	print('pred shape')
	print(prediction.shape)
	loss = criterion(prediction, target)
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	return loss.item()

loss_hist = []
for epoch in range(1):
	print("learning rate = {}".format(optimizer.param_groups[0]['lr']))
	for batch_idx, (data, target) in enumerate(train_dataloader):
		data = data.squeeze()
		data = data.unsqueeze(1)
		target = target.squeeze()
		target = target.unsqueeze(1)
		print('target shape')
		print(target.shape)
		loss = train_step(data, target)
		print("[{}:{}/{}]  LOSS={:.4}".format(epoch+1, batch_idx, len(train_dataloader), loss))
		loss_hist.append(loss)

# save model
torch.save(model.state_dict(), os.path.join('/data_qnap/yifeis/spherical_cnn/test/Unet_160k_test_final.pkl'))
# plot the loss and save the plot
print(len(loss_hist))
plt.plot(loss_hist)
plt.title("The Loss of the Test Model")
plt.savefig('/data_qnap/yifeis/spherical_cnn/test/Unet_160k_test_loss.png')
plt.clf()
