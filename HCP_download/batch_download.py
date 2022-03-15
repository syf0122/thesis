#useful resource for aws:
# AWS S3 high level api:
#   https://docs.aws.amazon.com/cli/latest/userguide/cli-services-s3-commands.html

# search for CHANGE for locations of where to customize for new files

import os
import threading
import _thread as thread
import multiprocessing
from multiprocessing import Process

with open('subjects.txt') as f:
    lines = f.readlines()

subjects = []
for l in lines:
    subjects.append(l.strip())

loc_dir = '/data_qnap/yifeis/HCP_3T/'

chunks = [subjects[x:x+35] for x in range(0, len(subjects), 35)]
print(len(chunks))

def download_data (sub_ls):
    for sub in sub_ls:
        os.system('mkdir -p ' + loc_dir + sub)
        rst1_dir = 's3://hcp-openaccess/HCP_1200/'+sub+'/MNINonLinear/fsaverage_LR32k/'+sub+'.L.midthickness_MSMAll.32k_fs_LR.surf.gii'
        save_dir = loc_dir + sub
        command_1 = 'aws s3 cp ' + rst1_dir + ' ' + save_dir + ' --region us-east-1'
        print(command_1)
        os.system(command_1)

        rst2_dir = 's3://hcp-openaccess/HCP_1200/'+sub+'/MNINonLinear/fsaverage_LR32k/'+sub+'.R.midthickness_MSMAll.32k_fs_LR.surf.gii'
        save_dir = loc_dir + sub
        command_2 = 'aws s3 cp ' + rst2_dir + ' ' + save_dir + ' --region us-east-1'
        print(command_2)
        os.system(command_2)
processes = []
for ls in chunks:
    p = Process(target=download_data, args=(ls, ))
    processes.append(p)
for p in processes:
    p.start()
for p in processes:
    p.join()
