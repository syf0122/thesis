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

# get the data for training
num_of_sub = 20
num_of_epo = 5
hemi = 'right'
gifti = True
################################################################

class BrainSphere(torch.utils.data.Dataset):

	def __init__(self, train_dir):
		# load all rs data
		self.data = np.load(train_dir)

	def __getitem__(self, index):
		one_tp_data = self.data[:, index]
		one_tp_data = torch.from_numpy(one_tp_data).unsqueeze(1)
		# print(one_tp_data.shape) # (163438, 1)
		return one_tp_data, one_tp_data

	def __len__(self):
		return self.data.shape[1]

if gifti:
	print('Using Gifti Data')
print(f'Training with {num_of_sub} subjects resting state data and {num_of_epo} epochs for {hemi} hemisphere.')
if gifti:
	# gifti_data
	train_data_dir = '/data_qnap/yifeis/spherical_cnn/test/first_'+str(num_of_sub)+'_'+hemi+'_train_gifti_data.npy'
else:
	# vtk data
	train_data_dir = '/data_qnap/yifeis/spherical_cnn/test/first_'+str(num_of_sub)+'_'+hemi+'_train_data.npy'

print(train_data_dir)
train_dataset = BrainSphere(train_data_dir) # number of subjects * 4 * 900
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, pin_memory=True) # each sample (163438, 1)
print(f'Training dataset shape: {train_dataset.data.shape}')

# get the model
model = Unet_160k(in_ch=in_channels, out_ch=out_channels)
print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
model.cuda(cuda)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training
def train_step(data, target):
	model.train()
	data, target = data.cuda(cuda), target.cuda(cuda)
	prediction = model(data)
	loss = criterion(prediction, target)
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	return loss.item()

loss_hist = []
for epoch in range(num_of_epo):
    epo_loss = []
    print("learning rate = {}".format(optimizer.param_groups[0]['lr']))
    for batch_idx, (data, target) in enumerate(train_dataloader):
        data = data.squeeze()
        data = data.unsqueeze(1)
        target = target.squeeze()
        target = target.unsqueeze(1)
        loss = train_step(data, target)
        print("[{}:{}/{}]  LOSS={:.4}".format(epoch+1, batch_idx, len(train_dataloader), loss))
        epo_loss.append(loss)
    epo_loss = np.array(epo_loss).mean()
    loss_hist.append(epo_loss)

# save model
if gifti:
	torch.save(model.state_dict(), os.path.join('/data_qnap/yifeis/spherical_cnn/test/Unet_160k_test_gifti_'+str(num_of_sub)+'_'+hemi+'_final.pkl'))
else:
	torch.save(model.state_dict(), os.path.join('/data_qnap/yifeis/spherical_cnn/test/Unet_160k_test_'+str(num_of_sub)+'_'+hemi+'_final.pkl'))
# plot the loss and save the plot
print(len(loss_hist))
plt.plot(loss_hist)
plt.title("The Loss of the Test Model")
plt.ylabel('MSE')
plt.xlabel('Epoch')
if gifti:
	plt.savefig('/data_qnap/yifeis/spherical_cnn/test/Unet_160k_test_gifti_'+str(num_of_sub)+'_'+hemi+'_loss.png')
else:
	plt.savefig('/data_qnap/yifeis/spherical_cnn/test/Unet_160k_test_'+str(num_of_sub)+'_'+hemi+'_loss.png')
plt.clf()
