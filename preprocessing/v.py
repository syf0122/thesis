import numpy as np
import matplotlib.pyplot as plt
import sys

# fig, axes = plt.subplots(nrows=1, ncols=4)

# subject = sys.argv[1]
data1 = np.load('Z:/data_qnap/yifeis/NAS/data/HC_p/sub-OAS30001_run-1_surf_norm_p.npy')
# data2 = np.load('/data_qnap/yifeis/new/processed/'+str(100610)+'/rest2_p.npy')
# data3 = np.load('/data_qnap/yifeis/new/processed/'+str(100610)+'/rest3_p.npy')
# data4 = np.load('/data_qnap/yifeis/new/processed/'+str(100610)+'/rest4_p.npy')


# distribution at single time timepoint
print(data1.shape)
mean_1 = np.mean(data1, axis = 0)
std_1 = np.std(data1, axis = 0)
data1 = data1.T
data1 = (data1 - np.mean(data1,axis=0))/np.std(data1,axis=0)
data1 = data1.T
mean_2 = np.mean(data1, axis = 0)
std_2 = np.std(data1, axis = 0)

plt.plot(mean_1, label = 'Normalized before parcellation')
plt.plot(mean_2, label = 'Further normalized in spatial domain')
plt.legend()
plt.title("Mean of 900 Timepoints")
plt.xlabel("Timepoints")
plt.ylabel("Mean")
plt.show()



plt.plot(std_1, label = 'Normalized before parcellation')
plt.plot(std_2, label = 'Further normalized in spatial domain')
plt.legend()
plt.title("Standard deviation of 900 Timepoints")
plt.xlabel("Timepoints")
plt.ylabel("std")
plt.show()
# # find minimum of minima & maximum of maxima
# minmin = np.min([np.min(data1), np.min(data2), np.min(data3), np.min(data4)])
# maxmax = np.max([np.max(data1), np.max(data2), np.max(data3), np.max(data4)])
# im1 = axes[0].imshow(data1, vmin=minmin, vmax=maxmax,
#                      extent=(-5,5,-5,5), aspect='auto', cmap='viridis')
# im2 = axes[1].imshow(data2, vmin=minmin, vmax=maxmax,
#                      extent=(-5,5,-5,5), aspect='auto', cmap='viridis')
# im3 = axes[2].imshow(data3, vmin=minmin, vmax=maxmax,
#                      extent=(-5,5,-5,5), aspect='auto', cmap='viridis')
# im4 = axes[3].imshow(data4, vmin=minmin, vmax=maxmax,
#                      extent=(-5,5,-5,5), aspect='auto', cmap='viridis')
#
# # add space for colour bar
# fig.subplots_adjust(right=0.85)
# cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
# fig.colorbar(im2, cax=cbar_ax)


# fig2, axes2 = plt.subplots(nrows=1, ncols=4)
#
# subject = 100610
# data1 = (data1 - data1.min()) / (data1.max() - data1.min())
# data2 = (data2 - data2.min()) / (data2.max() - data2.min())
# data3 = (data3 - data3.min()) / (data3.max() - data3.min())
# data4 = (data4 - data4.min()) / (data4.max() - data4.min())
#
# # find minimum of minima & maximum of maxima
# minmin = np.min([np.min(data1), np.min(data2), np.min(data3), np.min(data4)])
# maxmax = np.max([np.max(data1), np.max(data2), np.max(data3), np.max(data4)])
#
# im1 = axes2[0].imshow(data1, vmin=minmin, vmax=maxmax,
#                      extent=(-5,5,-5,5), aspect='auto', cmap='viridis')
# im2 = axes2[1].imshow(data2, vmin=minmin, vmax=maxmax,
#                      extent=(-5,5,-5,5), aspect='auto', cmap='viridis')
# im3 = axes2[2].imshow(data3, vmin=minmin, vmax=maxmax,
#                      extent=(-5,5,-5,5), aspect='auto', cmap='viridis')
# im4 = axes2[3].imshow(data4, vmin=minmin, vmax=maxmax,
#                      extent=(-5,5,-5,5), aspect='auto', cmap='viridis')
#
# # add space for colour bar
# fig2.subplots_adjust(right=0.85)
# cbar_ax = fig2.add_axes([0.88, 0.15, 0.04, 0.7])
# fig2.colorbar(im2, cax=cbar_ax)
# plt.show()
