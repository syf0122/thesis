import os
from tqdm import tqdm
import numpy as np

# # get all se results
# se_files = os.listdir('/data_qnap/yifeis/spherical_cnn/results/SE/')
# se_files.sort()
# progress_bar = tqdm(range(len(se_files)))
# for f in se_files:
#     if f.endswith('.npy'):
#         data = np.load('/data_qnap/yifeis/spherical_cnn/results/SE/' + f)
#         data = data.mean(axis=0)
#         np.save('/data_qnap/yifeis/spherical_cnn/results/SE/avg/' + f[:-4] + "_avg.npy", data)
#         progress_bar.update(1)
# progress_bar.close()


# ------------ (163842, Timepoints) ------------ #
# movie avg across all
mv_files = os.listdir('/data_qnap/yifeis/spherical_cnn/results/gifti_SE_5/')
mv_files.sort()
# m1 = np.zeros((163842, 921))
# m2 = np.zeros((163842, 918))
# m3 = np.zeros((163842, 915))
# m4 = np.zeros((163842, 901))
# progress_bar = tqdm(range(len(mv_files)))
# for f in mv_files:
#     if f.endswith('.npy') and 'left' in f:
#         if 'movie1' in f:
#             m1 += np.load('/data_qnap/yifeis/spherical_cnn/results/gifti_SE_5/' + f)
#         elif 'movie2' in f:
#             m2 += np.load('/data_qnap/yifeis/spherical_cnn/results/gifti_SE_5/' + f)
#         elif 'movie3' in f:
#             m3 += np.load('/data_qnap/yifeis/spherical_cnn/results/gifti_SE_5/' + f)
#         elif 'movie4' in f:
#             m4 += np.load('/data_qnap/yifeis/spherical_cnn/results/gifti_SE_5/' + f)
#     progress_bar.update(1)
# progress_bar.close()
# m1_avg = m1 / 80
# m2_avg = m2 / 80
# m3_avg = m3 / 80
# m4_avg = m4 / 80
# print(m1_avg.shape)
# print(m2_avg.shape)
# print(m3_avg.shape)
# print(m4_avg.shape)
# np.save('/data_qnap/yifeis/spherical_cnn/results/gifti_SE_5/avg/All_vb_movie1_avg_left.npy', m1_avg)
# np.save('/data_qnap/yifeis/spherical_cnn/results/gifti_SE_5/avg/All_vb_movie2_avg_left.npy', m2_avg)
# np.save('/data_qnap/yifeis/spherical_cnn/results/gifti_SE_5/avg/All_vb_movie3_avg_left.npy', m3_avg)
# np.save('/data_qnap/yifeis/spherical_cnn/results/gifti_SE_5/avg/All_vb_movie4_avg_left.npy', m4_avg)

m1 = np.zeros((163842, 921))
m2 = np.zeros((163842, 918))
m3 = np.zeros((163842, 915))
m4 = np.zeros((163842, 901))
progress_bar = tqdm(range(len(mv_files)))
for f in mv_files:
    if f.endswith('.npy') and 'right' in f:
        if 'movie1' in f:
            m1 += np.load('/data_qnap/yifeis/spherical_cnn/results/gifti_SE_5/' + f)
        elif 'movie2' in f:
            m2 += np.load('/data_qnap/yifeis/spherical_cnn/results/gifti_SE_5/' + f)
        elif 'movie3' in f:
            m3 += np.load('/data_qnap/yifeis/spherical_cnn/results/gifti_SE_5/' + f)
        elif 'movie4' in f:
            m4 += np.load('/data_qnap/yifeis/spherical_cnn/results/gifti_SE_5/' + f)
    progress_bar.update(1)
progress_bar.close()
m1_avg = m1 / 80
m2_avg = m2 / 80
m3_avg = m3 / 80
m4_avg = m4 / 80
print(m1_avg.shape)
print(m2_avg.shape)
print(m3_avg.shape)
print(m4_avg.shape)
np.save('/data_qnap/yifeis/spherical_cnn/results/gifti_SE_5/avg/All_vb_movie1_avg_right.npy', m1_avg)
np.save('/data_qnap/yifeis/spherical_cnn/results/gifti_SE_5/avg/All_vb_movie2_avg_right.npy', m2_avg)
np.save('/data_qnap/yifeis/spherical_cnn/results/gifti_SE_5/avg/All_vb_movie3_avg_right.npy', m3_avg)
np.save('/data_qnap/yifeis/spherical_cnn/results/gifti_SE_5/avg/All_vb_movie4_avg_right.npy', m4_avg)
