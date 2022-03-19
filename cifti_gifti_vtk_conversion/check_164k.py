import nibabel as nib


f = nib.load("/data_qnap/yifeis/new/102311/MNINonLinear/102311.MyelinMap_BC_MSMAll.164k_fs_LR.dscalar.nii")
print(type(f))
print(f.shape)
