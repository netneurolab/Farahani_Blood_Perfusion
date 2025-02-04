"""

# Resample cerebral blood perfusion maps from 32k fsLR to 4k fsLR resolution.

## Output Files:
    down-sampled cortical data:
    4k_perfusion_all_subjects_dev.npy
    4k_perfusion_all_subjects_aging.npy

    down-sampled sub-cortical data:
    4k_subcortex_perfusion_all_subjects_dev.npy
    4k_subcortex_perfusion_all_subjects_aging.npy

"""

#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import os
import globals
import warnings
import scipy.io
import numpy as np
import pandas as pd
import nibabel as nib
from IPython import get_ipython
from neuromaps import transforms
from functions import save_gifti
from nilearn.image import resample_img
from globals import path_templates, path_wb_command
from globals import path_results, path_mri, path_info_sub, path_medialwall, path_mri_dev

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------

warnings.filterwarnings("ignore")
get_ipython().magic('reset -sf')

#------------------------------------------------------------------------------
# Load subject information - HCP-development and aging
#------------------------------------------------------------------------------

perfusion_file_name = 'pvcorr_perfusion_calib_Atlas_MSMAll'

age_df = pd.read_csv(path_info_sub + 'clean_data_info.csv')
num_subjects_age = len(age_df)
subject_id_age = age_df.src_subject_id

dev_df = pd.read_csv(path_info_sub + 'Dev_clean_data_info.csv')
num_subjects_dev = len(dev_df)
subject_id_dev = dev_df.src_subject_id

#------------------------------------------------------------------------------
# Load medial wall information
#------------------------------------------------------------------------------

mask_medial_wall = scipy.io.loadmat(path_medialwall + 'fs_LR_32k_medial_mask.mat')['medial_mask']
mask_medial_wall = mask_medial_wall.astype(np.float32)

#------------------------------------------------------------------------------
# Resample cortical tissue - HCP-development
#------------------------------------------------------------------------------

all_data_dev = np.empty((8004, num_subjects_dev))
all_data_dev[:] = np.nan
for s, subid in enumerate(subject_id_dev):
    print(s)
    path_data_blood = os.path.join(path_mri_dev,  subid + '_V1_MR/MNINonLinear/ASL/')
    perfusion_file = os.path.join(path_data_blood, perfusion_file_name + '.dscalar.nii')
    img_perfusion = nib.cifti2.load(perfusion_file)
    perfusion_cortical = img_perfusion.get_data()[:, :globals.num_cort_vertices_noMW]

    perfusion_cortical_32k = np.empty((globals.num_cort_vertices_withMW, 1))
    perfusion_cortical_32k[:] = np.nan
    perfusion_cortical_32k[(mask_medial_wall == 1).flatten()] = perfusion_cortical.T

    save_gifti(perfusion_cortical_32k[:globals.num_vertices_gifti], path_results + 'perfusion_cortical_32k_L')
    save_gifti(perfusion_cortical_32k[globals.num_vertices_gifti:], path_results + 'perfusion_cortical_32k_R')

    Left = transforms.fslr_to_fslr(path_results + 'perfusion_cortical_32k_L.func.gii',
                                   target_density ='4k',
                                   hemi = {'L'},
                                   method ='linear')
    all_data_dev[:4002, s] = Left[0].darrays[0].data
    
    Right = transforms.fslr_to_fslr(path_results + 'perfusion_cortical_32k_R.func.gii',
                                   target_density ='4k',
                                   hemi = {'R'},
                                   method ='linear')
    all_data_dev[4002:, s] = Right[0].darrays[0].data

np.save(path_results + '4k_perfusion_all_subjects_dev.npy', all_data_dev)

#------------------------------------------------------------------------------
# Resample cortical tissue - HCP-aging
#------------------------------------------------------------------------------

all_data_age = np.empty((8004, num_subjects_age))
all_data_age[:] = np.nan
for s, subid in enumerate(subject_id_age):
    print(s)
    path_data_blood = os.path.join(path_mri, 'HCP_ASL/' + subid + '_V1_MR/MNINonLinear/ASL/')
    perfusion_file = os.path.join(path_data_blood, perfusion_file_name + '.dscalar.nii')
    img_perfusion = nib.cifti2.load(perfusion_file)
    perfusion_cortical = img_perfusion.get_data()[:, :globals.num_cort_vertices_noMW]

    perfusion_cortical_32k = np.empty((globals.num_cort_vertices_withMW, 1))
    perfusion_cortical_32k[:] = np.nan
    perfusion_cortical_32k[(mask_medial_wall == 1).flatten()] = perfusion_cortical.T

    save_gifti(perfusion_cortical_32k[:globals.num_vertices_gifti], path_results + 'perfusion_cortical_32k_L')
    save_gifti(perfusion_cortical_32k[globals.num_vertices_gifti:], path_results + 'perfusion_cortical_32k_R')

    Left = transforms.fslr_to_fslr(path_results + 'perfusion_cortical_32k_L.func.gii',
                                   target_density ='4k',
                                   hemi = {'L'},
                                   method ='linear')
    all_data_age[:4002, s] = Left[0].darrays[0].data
    
    Right = transforms.fslr_to_fslr(path_results + 'perfusion_cortical_32k_R.func.gii',
                                   target_density ='4k',
                                   hemi = {'R'},
                                   method ='linear')
    all_data_age[4002:, s] = Right[0].darrays[0].data

np.save(path_results + '4k_perfusion_all_subjects_aging.npy', all_data_age)

#------------------------------------------------------------------------------
# Create a downsampled MNI template
#------------------------------------------------------------------------------

# Load the 2mm MNI template
template_2mm_path = 'Atlas_ROIs.2.nii.gz'
template_2mm_img = nib.load(path_templates + template_2mm_path)
template_2mm_data = template_2mm_img.get_fdata()

# Define the target affine for 4mm resolution
target_affine = template_2mm_img.affine.copy()
target_affine[:3, :3] *= 2 # Double the voxel size from 2mm to 4mm

# Resample to 4mm resolution
template_4mm_img = resample_img(template_2mm_img,
                                target_affine = target_affine,
                                interpolation = 'nearest')

# Save the resampled image
template_4mm_path = 'template_4mm.nii'
nib.save(template_4mm_img, path_templates + template_4mm_path)
template_4mm_data = template_4mm_img.dataobj

#------------------------------------------------------------------------------
# Resample sub-cortical tissue - HCP-development
#------------------------------------------------------------------------------

all_data_subcortex_dev = np.zeros((4013, num_subjects_dev))
for s, subid in enumerate(subject_id_dev):
    print(s)
    path_data_blood = os.path.join(path_mri_dev,  subid + '_V1_MR/MNINonLinear/ASL/')
    perfusion_file = os.path.join(path_data_blood, perfusion_file_name + '.dscalar.nii')

    command = path_wb_command + 'wb_command -cifti-separate ' + \
    perfusion_file + ' COLUMN -volume-all '+  path_results + 'temp.nii.gz'
    os.system(command)

    temp_subject_img = nib.load(path_results + 'temp.nii.gz')
    temp_subject_data = temp_subject_img.get_fdata()

    # Define the target affine for 4mm resolution
    target_affine = temp_subject_img.affine.copy()
    target_affine[:3, :3] *= 2  # Double the voxel size from 2mm to 4mm

    # Resample to 4mm resolution
    temp_subject_4mm_img = resample_img(temp_subject_img,
                                    target_affine = target_affine,
                                    interpolation = 'linear')
    temp_subject_4mm_data = temp_subject_4mm_img.dataobj
    all_data_subcortex_dev[:, s] = temp_subject_4mm_data[np.where(template_4mm_data != 0)]

np.save(path_results + '4k_subcortex_perfusion_all_subjects_dev.npy', all_data_subcortex_dev)

#------------------------------------------------------------------------------
# Resample sub-cortical tissue - HCP-aging
#------------------------------------------------------------------------------

all_data_subcortex_age = np.zeros((4013, num_subjects_age))
for s, subid in enumerate(subject_id_age):
    print(s)
    path_data_blood = os.path.join(path_mri, 'HCP_ASL/' + subid + '_V1_MR/MNINonLinear/ASL/')
    perfusion_file = os.path.join(path_data_blood, perfusion_file_name + '.dscalar.nii')

    command = path_wb_command + 'wb_command -cifti-separate ' + \
    perfusion_file + ' COLUMN -volume-all '+  path_results + 'temp.nii.gz'
    os.system(command)

    temp_subject_img = nib.load(path_results + 'temp.nii.gz')
    temp_subject_data = temp_subject_img.get_fdata()

    # Define the target affine for 4mm resolution
    target_affine = temp_subject_img.affine.copy()
    target_affine[:3, :3] *= 2  # Double the voxel size from 2mm to 4mm

    # Resample to 4mm resolution
    temp_subject_4mm_img = resample_img(temp_subject_img,
                                    target_affine = target_affine,
                                    interpolation = 'linear')
    temp_subject_4mm_data = temp_subject_4mm_img.dataobj
    all_data_subcortex_age[:, s] = temp_subject_4mm_data[np.where(template_4mm_data != 0)]

np.save(path_results + '4k_subcortex_perfusion_all_subjects_aging.npy', all_data_subcortex_age)

#------------------------------------------------------------------------------
# END