import numpy as np
import nibabel as nib

# 2mm atlas
atlas_2 = nib.load("/data_qnap/yifeis/NAS/data/mmp_atlas/HCP-MMP1_on_MNI152_ICBM2009a_nlin_2mm.nii.gz")
atlas_2_data = atlas_2.get_fdata()
print(atlas_2.shape) # (99, 117, 95) - (8, 8, 4)

atlas_2_data_cr = atlas_2_data[4:, 4:, 2:]
atlas_2_data_cr = atlas_2_data_cr[:(atlas_2_data_cr.shape[0] - 4), :(atlas_2_data_cr.shape[1] - 4), :(atlas_2_data_cr.shape[2] - 2)]
print(atlas_2_data_cr.shape) # (91, 109, 91)

# atlas_2_data_cr = np.rint(atlas_2_data_cr)
# cropped_img = nib.Nifti1Image(atlas_2_data_cr, atlas_2.affine)
# nib.save(cropped_img, '/data_qnap/yifeis/NAS/data/mmp_atlas/HCP-MMP1_on_MNI152_ICBM2009a_nlin_2mm_cropped.nii.gz')

# 4mm atlas
atlas_4 = nib.load("/data_qnap/yifeis/NAS/data/mmp_atlas/HCP-MMP1_on_MNI152_ICBM2009a_nlin_4mm.nii.gz")
atlas_4_data = atlas_4.get_fdata()
print(atlas_4_data.shape) # (50, 59, 48) + (1, 1, 1) -> (51, 60, 49)

atlas_4_data_cr = np.pad(atlas_4_data, (0, 1), 'edge')
print(atlas_4_data_cr.shape) # (51, 60, 49) - (2, 2, 0) -> (49, 58, 49)

atlas_4_data_cr = atlas_4_data_cr[:(atlas_4_data_cr.shape[0] - 2), :(atlas_4_data_cr.shape[1] - 2), :]
print(atlas_4_data_cr.shape) # (49, 58, 49)

# atlas_4_data_cr = np.rint(atlas_4_data_cr)
# cropped_img = nib.Nifti1Image(atlas_4_data_cr, atlas_4.affine)
# nib.save(cropped_img, '/data_qnap/yifeis/NAS/data/mmp_atlas/HCP-MMP1_on_MNI152_ICBM2009a_nlin_4mm_cropped.nii.gz')


atlas_2 = nib.load("/data_qnap/yifeis/NAS/data/mmp_atlas/HCP-MMP1_on_MNI152_ICBM2009a_nlin_2mm_cropped.nii.gz")
atlas_2_data = atlas_2.get_fdata()
atlas_4 = nib.load("/data_qnap/yifeis/NAS/data/mmp_atlas/HCP-MMP1_on_MNI152_ICBM2009a_nlin_4mm_cropped.nii.gz")
atlas_4_data = atlas_4.get_fdata()

print(atlas_2_data.shape)
print(atlas_4_data.shape)
