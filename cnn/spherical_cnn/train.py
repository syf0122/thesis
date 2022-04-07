"""
Adapted from Fenqiang Zhao, https://github.com/zhaofenqiang
"""

import torch
import torch.nn as nn
import torchvision
import scipy.io as sio
import numpy as np
import glob
import os

from sphericalunet.utils.utils import compute_weight
from sphericalunet.utils.vtk import read_vtk, write_vtk, resample_label
import matplotlib.pyplot as plt

################################################################
""" hyper-parameters """
cuda = torch.device('cuda:0')
batch_size = 1
in_channels = 1
out_channels = 1
learning_rate = 0.001
################################################################

class BrainSphere(torch.utils.data.Dataset):

    def __init__(self, root, subjects):
        for s in subjects:
            self.files = self.files + sorted(glob.glob(os.path.join(root, s, '/*.vtk')))

    def __getitem__(self, index):
        file = self.files[index]
        data = read_vtk(file)
        tseries_data = orig_data['cmap']
        tseries_data = torch.from_numpy(tseries_data).unsqueeze(1)
        print(tseries_data.shape)
        # target = input for reconstruction
        return tseries_data.astype(np.float64), tseries_data.astype(np.float64)

    def __len__(self):
        return len(self.files)

dir = '/data_qnap/yifeis/HCP_7T/'
subs = os.listdir(dir)
sub.sort()

for fo in range(5):
    test_subs  = subs[fo*10: fo*10 + 10]
    train_subs = []
    for s in subs:
        if s not in test_subs:
            train_subs.append(s)

    # get the data for training and testing
    train_dataset = BrainSphere(dir, train_subs)
    test_dataset = BrainSphere(dir, test_subs)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    model = Unet_160k(in_ch=in_channels, out_ch=out_channels)
    print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    model.cuda(cuda)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.2, patience=1, verbose=True, threshold=0.0001, threshold_mode='rel', min_lr=0.000001)

    def train_step(data, target):
        model.train()
        data, target = data.cuda(cuda), target.cuda(cuda)
        prediction = model(data)
        loss = criterion(prediction, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()


    # def compute_dice(pred, gt):
    #     pred = pred.cpu().numpy()
    #     gt = gt.cpu().numpy()
    #     dice = np.zeros(36)
    #     for i in range(36):
    #         gt_indices = np.where(gt == i)[0]
    #         pred_indices = np.where(pred == i)[0]
    #         dice[i] = 2 * len(np.intersect1d(gt_indices, pred_indices))/(len(gt_indices) + len(pred_indices))
    #     return dice


    # def val_during_training(dataloader):
    #     model.eval()
    #     dice_all = np.zeros((len(dataloader),36))
    #     for batch_idx, (data, target) in enumerate(dataloader):
    #         data = data.squeeze()
    #         target = target.squeeze()
    #         data, target = data.cuda(cuda), target.cuda(cuda)
    #         with torch.no_grad():
    #             prediction = model(data)
    #         dice_all[batch_idx,:] = compute_dice(prediction, target)
    #
    #     return dice_all


    # train_dice = [0, 0, 0, 0, 0]
    loss_hist = []
    for epoch in range(100):

        # train_dc = val_during_training(train_dataloader)
        # print("train Dice: ", np.mean(train_dc, axis=0))
        # print("train_dice, mean, std: ", np.mean(train_dc), np.std(np.mean(train_dc, 1)))
        #
        # val_dc = val_during_training(val_dataloader)
        # print("val Dice: ", np.mean(val_dc, axis=0))
        # print("val_dice, mean, std: ", np.mean(val_dc), np.std(np.mean(val_dc, 1)))
        # writer.add_scalars('data/Dice', {'train': np.mean(train_dc), 'val':  np.mean(val_dc)}, epoch)
        #
        # scheduler.step(np.mean(val_dc))

        print("learning rate = {}".format(optimizer.param_groups[0]['lr']))

        for batch_idx, (data, target) in enumerate(train_dataloader):
            data = data.squeeze()
            target = target.squeeze()
            loss = train_step(data, target)
            print("[{}:{}/{}]  LOSS={:.4}".format(epoch,
                  batch_idx, len(train_dataloader), loss))
            writer.add_scalar('Train/Loss', loss, epoch*len(train_dataloader) + batch_idx)
            loss_hist.append(loss)
        # train_dice[epoch % 5] = np.mean(train_dc)
        # print("last five train Dice: ",train_dice)
        # if np.std(np.array(train_dice)) <= 0.00001:
        #     torch.save(model.state_dict(), os.path.join('trained_models', model_name+'_'+str(fold)+"_final.pkl"))
        #     break
        # torch.save(model.state_dict(), os.path.join('trained_models', model_name+'_'+str(fold)+".pkl"))

    # save model
    torch.save(model.state_dict(), os.path.join('/data_qnap/yifeis/spherical_cnn/models/', model_name+'_'+str(fo)+"_final.pkl"))
    # plot the loss and save the plot
    plt.plot(loss)
    plt.title("The Loss of the Model in Fold " + str(fo))
    plt.save_fig('/data_qnap/yifeis/spherical_cnn/models/' + model_name+'_'+str(fo)+"_loss.png")
    plt.clf()
