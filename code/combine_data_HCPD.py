"""

Load developmental data (perfusion, arrival, thickness, and myelin), combine them, and save the results

The generated outputs are named:
    Dev_arrival_all_vertex.npy
    Dev_perfusion_all_vertex.npy
    Dev_thickness_all_vertex.npy
    Dev_myelin_all_vertex.npy

We also calculate the mean of the data across all subjects:
    Dev_arrival_mean_across_subjects_vertexwise.dscalar.nii
    Dev_perfusion_mean_across_subjects_vertexwise.dscalar.nii
    Dev_thickness_mean_across_subjects_vertexwise.dscalar.nii
    Dev_myelin_mean_across_subjects_vertexwise.dscalar.nii

"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import os
import glob
import globals
import warnings
import numpy as np
import pandas as pd
import nibabel as nib
from IPython import get_ipython
from functions import save_as_dscalar_and_npy
from globals import path_results, path_info_sub, path_mri_dev, path_mri

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------

warnings.filterwarnings("ignore")
get_ipython().magic('reset -sf')

#------------------------------------------------------------------------------
# Get subject list - whose data is available?
#------------------------------------------------------------------------------

subject_ids = []
subject_names = []
for subject_folder in glob.glob(os.path.join(path_mri_dev, "HCD*_V1_MR")):
    subject_name = os.path.basename(subject_folder)
    subject_ids.append(subject_name)
    subject_names.append(subject_name[:-6])

num_subjects = len(subject_ids)

#------------------------------------------------------------------------------
# Save a clean version of subject information dataframe
#------------------------------------------------------------------------------

df = pd.read_csv(path_info_sub + 'HCD_LS_2.0_subject_completeness.csv')
df = df.drop(0).reset_index(drop = True)
df_filtered = df[df['src_subject_id'].isin(subject_names)]
df_filtered = df_filtered.reset_index(drop = True)
df_filtered["interview_age"] = pd.to_numeric(df_filtered["interview_age"])

# This is the DataFrame that will be used in later scripts 
df_filtered.to_csv(path_info_sub + 'Dev_clean_data_info.csv')

# Extract clean subject IDs
subject_ids_clean = df_filtered.src_subject_id

#------------------------------------------------------------------------------
# Load vertex-wise data, combine them, and save the combined data.
#------------------------------------------------------------------------------

# Initialize arrays for combined data
arrival_all_vertexwise   = np.zeros((globals.num_vertices_voxels, num_subjects))
perfusion_all_vertexwise = np.zeros((globals.num_vertices_voxels, num_subjects))
thickness_all_vertexwise = np.zeros((globals.num_cort_vertices_noMW, num_subjects))
myelin_all_vertexwise    = np.zeros((globals.num_cort_vertices_noMW, num_subjects))

for n, subid in enumerate(subject_ids_clean):

    # Define paths for ASL and morph data (etc. thickness)
    path_data_ASL = os.path.join(path_mri_dev,  subid + '_V1_MR/MNINonLinear/ASL/')
    path_data_morph = os.path.join(path_mri, 'HCP_morph_development/' + subid + '_V1_MR/MNINonLinear/fsaverage_LR32k/')

    # Load arterial arrival time (ATT) data
    file_arrival = os.path.join(path_data_ASL + 'pvcorr_arrival_Atlas_MSMAll.dscalar.nii')
    img_arrival = nib.cifti2.load(file_arrival)
    arrival_all_vertexwise[:, n] = img_arrival.get_data()

    # Load blood perfusion data (CBF)
    file_per = os.path.join(path_data_ASL + 'pvcorr_perfusion_calib_Atlas_MSMAll.dscalar.nii')
    img_per = nib.cifti2.load(file_per)
    perfusion_all_vertexwise[:, n] = img_per.get_data()

    # Load cortical thickness data (estimated based on T1 & T2 data)
    file_th = os.path.join(path_data_morph + subid + '_V1_MR.thickness_MSMAll.32k_fs_LR.dscalar.nii')
    img_th = nib.cifti2.load(file_th)
    thickness_all_vertexwise[:, n] = img_th.get_data()

    # Load myelin data (estimated based on T1 & T2 data)
    file_myelin = os.path.join(path_data_morph + subid + '_V1_MR.SmoothedMyelinMap_BC_MSMAll.32k_fs_LR.dscalar.nii')
    img_myelin = nib.cifti2.load(file_myelin)
    myelin_all_vertexwise[:, n] = img_myelin.get_data()

    print(n) # To show the progress of the process

# Save combined vertex-wise data
np.save(path_results + 'Dev_arrival_all_vertex.npy', arrival_all_vertexwise)
np.save(path_results + 'Dev_perfusion_all_vertex.npy', perfusion_all_vertexwise)
np.save(path_results + 'Dev_thickness_all_vertex.npy', thickness_all_vertexwise)
np.save(path_results + 'Dev_myelin_all_vertex.npy', myelin_all_vertexwise)

#------------------------------------------------------------------------------
# Load voxel-wise data that have been generated & save their mean across subjects
#------------------------------------------------------------------------------

data_files = ['arrival', 'perfusion', 'thickness', 'myelin']
data_vertexwise = {name: np.load(path_results + f"Dev_{name}_all_vertex.npy") for name in data_files}

for name, data in data_vertexwise.items():
    type_data = 'cortex' if name in ['myelin', 'thickness'] else 'cortex_subcortex'
    data_to_save = np.mean(data, axis = 1)
    save_as_dscalar_and_npy(data_to_save, type_data, path_results , f'Dev_{name}_mean_across_subjects_vertexwise')
    del data_to_save

#------------------------------------------------------------------------------
# END