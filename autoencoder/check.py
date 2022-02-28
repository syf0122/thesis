import nibabel as nib
import hcp_utils as hcp


vol_data = nib.load("/data_qnap/yifeis/new/processed/100610/temporal_filtered_tfMRI_MOVIE1_7T_AP_Atlas_MSMAll_hp2000_clean.dtseries.nii")
vol_data = vol_data.get_fdata()
vol_data = hcp.normalize(vol_data)
vol_df = pd.DataFrame(vol_data)

latent_data_dir[sub].append("/data_qnap/yifeis/ae_latent/"+dir)
lat_df = pd.read_csv("/data_qnap/yifeis/ae_latent/",index_col=0)
