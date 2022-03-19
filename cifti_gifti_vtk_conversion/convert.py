import os
import numpy as np
import nibabel as nib
from nibabel.testing import test_data

subjects = os.listdir('/data_qnap/yifeis/new/processed/')
subjects.sort()
subjects = subjects[-1:]
print(len(subjects))
print(subjects)
for sub in subjects:
    save_dir = '/data_qnap/yifeis/HCP_7T/'+sub
    # print('mkdir -p ' + save_dir)
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

    temp_164k_dir = '/data_qnap/yifeis/new/'+sub+'/MNINonLinear/'+sub+'.thickness_MSMAll.164k_fs_LR.dscalar.nii'
    left_sphere_32k_dir = '/data_qnap/yifeis/new/'+sub+'/MNINonLinear/fsaverage_LR32k/'+sub+'.L.sphere.32k_fs_LR.surf.gii'
    left_sphere_164k_dir = '/data_qnap/yifeis/new/'+sub+'/MNINonLinear/'+sub+'.L.sphere.164k_fs_LR.surf.gii'
    left_area_32k_dir = '/data_qnap/yifeis/new/'+sub+'/MNINonLinear/fsaverage_LR32k/'+sub+'.L.midthickness_MSMAll.32k_fs_LR.surf.gii'
    left_area_164k_dir = '/data_qnap/yifeis/new/'+sub+'/MNINonLinear/'+sub+'.L.midthickness_MSMAll.164k_fs_LR.surf.gii'

    right_sphere_32k_dir = '/data_qnap/yifeis/new/'+sub+'/MNINonLinear/fsaverage_LR32k/'+sub+'.R.sphere.32k_fs_LR.surf.gii'
    right_sphere_164k_dir = '/data_qnap/yifeis/new/'+sub+'/MNINonLinear/'+sub+'.R.sphere.164k_fs_LR.surf.gii'
    right_area_32k_dir = '/data_qnap/yifeis/new/'+sub+'/MNINonLinear/fsaverage_LR32k/'+sub+'.R.midthickness_MSMAll.32k_fs_LR.surf.gii'
    right_area_164k_dir = '/data_qnap/yifeis/new/'+sub+'/MNINonLinear/'+sub+'.R.midthickness_MSMAll.164k_fs_LR.surf.gii'
    for ses in sessions:
        for f in files:
            if ses.upper() in f and 'temporal_filtered_' in f and '.dtseries.nii' in f:
                cifti_dir = '/data_qnap/yifeis/new/processed/'+sub+'/'+f
                # print(cifti_dir)
                cifti_164k_dir = save_dir+'/'+ses+'_164k.dtseries.nii'
                # print(cifti_164k_dir)
        # upsampling
        command_1 = 'wb_command -cifti-resample ' + cifti_dir + ' COLUMN ' \
                                                  + temp_164k_dir + ' COLUMN ADAP_BARY_AREA ENCLOSING_VOXEL ' \
                                                  + cifti_164k_dir + ' -left-spheres ' \
                                                  + left_sphere_32k_dir + ' ' \
                                                  + left_sphere_164k_dir + ' -left-area-surfs ' \
                                                  + left_area_32k_dir + ' ' \
                                                  + left_area_164k_dir + ' -right-spheres ' \
                                                  + right_sphere_32k_dir + ' ' \
                                                  + right_sphere_164k_dir + ' -right-area-surfs ' \
                                                  + right_area_32k_dir + ' ' \
                                                  + right_area_164k_dir
        command_2 = 'wb_command -cifti-separate ' + cifti_164k_dir + ' COLUMN -metric CORTEX_LEFT ' + save_dir + '/' + ses +'_left.func.gii -metric CORTEX_RIGHT ' +save_dir + '/' + ses + '_right.func.gii'
        print(command_1)
        os.system(command_1)
        print(command_2)
        os.system(command_2)


# # check the template
# sub = '100610'
# temp_164k_dir = '/data_qnap/yifeis/new/'+sub+'/MNINonLinear/'+sub+'.thickness_MSMAll.164k_fs_LR.dscalar.nii'
# r1_164k_dir = '/data_qnap/yifeis/HCP_7T/'+sub+'/rest1_164k.dtseries.nii'
# data = nib.load(temp_164k_dir)
# r1 = nib.load(r1_164k_dir)
# print(data.shape)
# print(r1.shape)

# # read gifti
# sub = '100610'
# gdata = nib.load('/data_qnap/yifeis/HCP_7T/'+sub+'/rest1_left.func.gii')
# coords = gdata.agg_data()
# print(len(coords))
# print(len(coords[0]))
