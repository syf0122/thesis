#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import nibabel as nib


# # crop 2mm atlas

# In[8]:


atlas_2 = nib.load("HCP-MMP1_on_MNI152_ICBM2009a_nlin_2mm.nii.gz")
atlas_2_data = atlas_2.get_fdata()
print(atlas_2.shape)


# In[9]:


atlas_2_data_cr = atlas_2_data[4:, 4:, 2:]
atlas_2_data_cr = atlas_2_data_cr[:(atlas_2_data_cr.shape[0] - 4), :(atlas_2_data_cr.shape[1] - 4), :(atlas_2_data_cr.shape[2] - 2)]
atlas_2_data_cr.shape


# In[10]:


atlas_2_data_cr = np.rint(atlas_2_data_cr)
cropped_img = nib.Nifti1Image(atlas_2_data_cr, atlas_2.affine)
nib.save(cropped_img, 'HCP-MMP1_on_MNI152_ICBM2009a_nlin_2mm_cropped.nii.gz')  


# In[11]:


# atlas_2_data_cr = atlas_2_data_cr.astype(int)
np.unique(atlas_2_data_cr)


# # Crop 4mm atlas

# In[12]:


atlas_4 = nib.load("HCP-MMP1_on_MNI152_ICBM2009a_nlin_4mm.nii.gz")
atlas_4_data = atlas_4.get_fdata()
print(atlas_4_data.shape)


# In[70]:


atlas_4_data_cr = np.pad(atlas_4_data, (0, 1), 'edge')
atlas_4_data_cr.shape


# In[71]:


atlas_4_data_cr = atlas_4_data_cr[:(atlas_4_data_cr.shape[0] - 2), :(atlas_4_data_cr.shape[1] - 2), :]
atlas_4_data_cr.shape


# In[72]:


atlas_4_data_cr = np.rint(atlas_4_data_cr)
cropped_img = nib.Nifti1Image(atlas_4_data_cr, atlas_4.affine)
nib.save(cropped_img, 'HCP-MMP1_on_MNI152_ICBM2009a_nlin_4mm_cropped.nii.gz')  


# In[73]:


np.unique(atlas_4_data_cr)


# In[ ]:




