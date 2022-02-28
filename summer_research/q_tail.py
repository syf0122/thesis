import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

hc_quality = pd.read_csv('/data_qnap/yifeis/NAS/data/quality/HC_mean_param.csv', index_col=0)
ad_quality = pd.read_csv('/data_qnap/yifeis/NAS/data/quality/AD_mean_param.csv', index_col=0)
print("The total number of sesssions: HC -- {}, AD -- {}".format(hc_quality.shape[0], ad_quality.shape[0]))

# rank in desc order
hc_dvars_sorted = hc_quality.sort_values(by ='std_dvars')
hc_fd_sorted = hc_quality.sort_values(by ='framewise_displacement')
ad_dvars_sorted = ad_quality.sort_values(by ='std_dvars')
ad_fd_sorted = ad_quality.sort_values(by ='framewise_displacement')

# get the highest 5%
hc_5_perc = round(hc_quality.shape[0] * 0.05)
ad_5_perc = round(ad_quality.shape[0] * 0.05)

hc_high_dvars = hc_dvars_sorted.head(hc_5_perc)
hc_high_fd = hc_fd_sorted.head(hc_5_perc)
ad_high_dvars = ad_dvars_sorted.head(ad_5_perc)
ad_high_fd = ad_fd_sorted.head(ad_5_perc)
print("The number of highest 5% sesssions: HC -- {}, AD -- {}".format(hc_high_dvars.shape[0], ad_high_fd.shape[0]))

hc_high_dvars.to_csv('/data_qnap/yifeis/NAS/data/quality/HC_highest_5%_std_dvars.csv')
hc_high_fd.to_csv('/data_qnap/yifeis/NAS/data/quality/HC_highest_5%_fd.csv')
ad_high_dvars.to_csv('/data_qnap/yifeis/NAS/data/quality/AD_highest_5%_std_dvars.csv')
ad_high_fd.to_csv('/data_qnap/yifeis/NAS/data/quality/AD_highest_5%_fd.csv')

# save the other 95% dir
hc_low_dvars = hc_dvars_sorted.tail(hc_quality.shape[0] - hc_5_perc)
hc_low_fd = hc_fd_sorted.tail(hc_quality.shape[0] - hc_5_perc)
ad_low_dvars = ad_dvars_sorted.tail(ad_quality.shape[0]- ad_5_perc)
ad_low_fd = ad_fd_sorted.tail(ad_quality.shape[0]- ad_5_perc)
print("The number of lower 95% sesssions: HC -- {}, AD -- {}".format(hc_low_dvars.shape[0], ad_low_fd.shape[0]))

# save the dir
hc_filtered_dvars_dir = []
hc_filtered_fd_dir = []
ad_filtered_dvars_dir = []
ad_filtered_fd_dir = []
hc_dvars_subs = []
hc_fd_subs = []
ad_dvars_subs = []
ad_fd_subs = []
for i in hc_low_dvars.index:
    sub = i[:12]
    if sub not in hc_dvars_subs:
        hc_dvars_subs.append(sub)
    ses = i[13:22]
    file_dir =  '/data_qnap/yifeis/NAS/data/HC/'+sub+'/'+ses+'/func/'+i
    hc_filtered_dvars_dir.append(file_dir)
for i in hc_low_fd.index:
    sub = i[:12]
    if sub not in hc_fd_subs:
        hc_fd_subs.append(sub)
    ses = i[13:22]
    file_dir =  '/data_qnap/yifeis/NAS/data/HC/'+sub+'/'+ses+'/func/'+i
    hc_filtered_fd_dir.append(file_dir)
for i in ad_low_dvars.index:
    sub = i[:12]
    if sub not in ad_dvars_subs:
        ad_dvars_subs.append(sub)
    ses = i[13:22]
    file_dir =  '/data_qnap/yifeis/NAS/data/HC/'+sub+'/'+ses+'/func/'+i
    ad_filtered_dvars_dir.append(file_dir)
for i in ad_low_fd.index:
    sub = i[:12]
    if sub not in ad_fd_subs:
        ad_fd_subs.append(sub)
    ses = i[13:22]
    file_dir =  '/data_qnap/yifeis/NAS/data/HC/'+sub+'/'+ses+'/func/'+i
    ad_filtered_fd_dir.append(file_dir)
hc_dir_df = pd.DataFrame()
ad_dir_df = pd.DataFrame()
hc_dir_df['dvars_low_ses'] = hc_filtered_dvars_dir
hc_dir_df['fd_low_ses'] = hc_filtered_fd_dir
ad_dir_df['dvars_low_ses'] = ad_filtered_dvars_dir
ad_dir_df['fd_low_ses'] = ad_filtered_fd_dir
hc_dir_df.to_csv('/data_qnap/yifeis/NAS/data/quality/HC_better_quality_sessions_dir.csv')
ad_dir_df.to_csv('/data_qnap/yifeis/NAS/data/quality/AD_better_quality_sessions_dir.csv')

# total number of subjects:
print("The total number of subjects: HC -- {}, AD -- {}".format(len(os.listdir('/data_qnap/yifeis/NAS/data/HC/')), len(os.listdir('/data_qnap/yifeis/NAS/data/AD/'))))
print("The number of HC subjects with lower: DVARS -- {}, FD -- {}".format(len(hc_dvars_subs), len(hc_fd_subs)))
print("The number of AD subjects with lower: DVARS -- {}, FD -- {}".format(len(ad_dvars_subs), len(ad_fd_subs)))

# save the intersect
hc_inter = np.intersect1d(hc_low_dvars.index, hc_low_fd.index)
ad_inter = np.intersect1d(ad_low_dvars.index, ad_low_fd.index)
hc_subs = []
ad_subs = []
for i in hc_inter:
    sub = i[:12]
    if sub not in hc_subs:
        hc_subs.append(sub)
for i in ad_inter:
    sub = i[:12]
    if sub not in ad_subs:
        ad_subs.append(sub)
print("The number of sessions with lower DVARS and FD: HC -- {}, AD -- {}".format(len(hc_inter), len(ad_inter)))
print("The number of subjects with lower DVARS and FD: HC -- {}, AD -- {}".format(len(hc_subs), len(ad_subs)))
hc_inter_df = pd.DataFrame()
ad_inter_df = pd.DataFrame()
hc_inter_df['Intersect'] = hc_inter
ad_inter_df['Intersect'] = ad_inter
hc_inter_df.to_csv('/data_qnap/yifeis/NAS/data/quality/HC_better_quality_intersect_dir.csv')
ad_inter_df.to_csv('/data_qnap/yifeis/NAS/data/quality/AD_better_quality_intersect_dir.csv')
