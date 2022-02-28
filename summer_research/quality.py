
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

gp = 'AD'
dir = '/data_qnap/yifeis/NAS/data/'+gp+'/'
save_dir = '/data_qnap/yifeis/NAS/data/quality/'+gp+'_mean_param.csv'

# get all subjects
subjects = os.listdir(dir)
subjects.sort()
print("There are {} {} subjects".format(len(subjects)， gp))

# get directories of files and load data
tsv_data = []
sessions = []
progress_bar = tqdm(total = len(subjects))

for subject in subjects:
    # get the directory of to the functional data of this subject
    subject_dir = os.listdir(dir+subject)
    func_dir = None
    for d in subject_dir:
        if "ses-" in d:
            func_dir = dir+subject+"/"+d+"/func/"

    if func_dir == None:
        print("No such directory for subject " + subject)
    else:
        func_files = os.listdir(func_dir)
        func_files.sort()
        for f in func_files:
            # find smoothed data
            if "_desc-confounds_timeseries.tsv" in f:
                df = pd.read_csv(func_dir+f, sep='\t')    # load data
                df = df[['std_dvars', 'dvars', 'framewise_displacement']]
                tsv_data.append(df)
                sessions.append(f[:-30]) # save the count of run
    progress_bar.update(1)
progress_bar.close()
print("Data loaded!")
print(len(sessions))
for n in tsv_data:
    if n.shape != (164, 3):
        print(n.shape)
print(len(sessions) == len(tsv_data))

# calculated the mean
avg_df = pd.DataFrame(columns=['std_dvars', 'dvars', 'framewise_displacement'], index=sessions)
for i in range(len(tsv_data)):
    df = tsv_data[i]
    row_idx = sessions[i]
    df = df.mean(axis=0)
    avg_df.loc[row_idx] = [df['std_dvars'], df['dvars'], df['framewise_displacement']]
avg_df.to_csv(save_dir)
