import os
import numpy as np
import nibabel as nib
from nibabel.testing import test_data

subjects = os.listdir('/data_qnap/yifeis/new/processed/')
subjects.sort()

for sub in subjects:
    save_dir = '/data_qnap/yifeis/HCP_7T/'+sub
    print('mkdir -p ' + save_dir)
    os.system('mkdir -p ' + save_dir)
    files = os.listdir('/data_qnap/yifeis/new/processed/'+sub+'/')
    files.sort()
    sessions = ['rest1',
                'rest2',
                'rest3',
                'rest4',
                'movie1',
                'movie2',
                'movie3',
                'movie4',
                'retbar1',
                'retbar2',
                'retccw',
                'retcon',
                'retcw',
                'retexp']
    for ses in sessions:
        for f in files:
            if ses.upper() in f and 'temporal_filtered_' in f and '.dtseries.nii' in f:
                cifti_dir = '/data_qnap/yifeis/new/processed/'+sub+'/'+f
                print(cifti_dir)
        command = 'wb_command -cifti-separate ' +cifti_dir+ ' COLUMN -metric CORTEX_LEFT '+save_dir+'/'+ses+'_left.func.gii -metric CORTEX_RIGHT '+save_dir+'/'+ses+'_right.func.gii'
        print(command)
        os.system(command)


# # read gifti
# gdata = nib.load("/home/yifeis/code/conversion/left_scalar.func.gii")
# coords = gdata.agg_data()
# print(len(coords))
# # print(len(coords[0]))
