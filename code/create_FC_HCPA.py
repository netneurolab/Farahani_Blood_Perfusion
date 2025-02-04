"""

# Calculate Functional Connectivity for Subjects with Perfusion Data - HCP Aging

This script computes the functional connectivity (FC) for subjects within the HCP Aging dataset who have associated perfusion data.
The process involves several key steps to ensure accurate and meaningful FC matrices are generated for each subject:

    1. Load and demean vertex-wise time-series
    2. Parcellate data using Schaefer-400 atlas
    3. Z-score parcel-wise time-series
    4. Concatenate normalized time-series across runs
    5. Compute Pearson correlation as functional connectivity

**Note**
- **HCP-Aging dataset structure:**
    All subjects have 4 runs, provided their data is complete.

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
from scipy.stats import zscore
from IPython import get_ipython
from neuromaps.images import load_data
from neuromaps.images import dlabel_to_gifti
from netneurotools.datasets import fetch_schaefer2018
from globals import path_FC, path_info_sub, path_medialwall, path_mri

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------

warnings.filterwarnings("ignore")
get_ipython().magic('reset -sf')

#------------------------------------------------------------------------------
# Get subject list
#------------------------------------------------------------------------------

df = pd.read_csv(path_info_sub + 'clean_data_info.csv')
num_subjects = len(df)
subject_ids = df.src_subject_id

#------------------------------------------------------------------------------
# Load subject specific data and process them
#------------------------------------------------------------------------------

schaefer = fetch_schaefer2018('fslr32k')[f"{globals.nnodes_Schaefer}Parcels7Networks"]
atlas = load_data(dlabel_to_gifti(schaefer))
mask_medial_wall = scipy.io.loadmat(path_medialwall + 'fs_LR_32k_medial_mask.mat')['medial_mask']
mask_medial_wall = mask_medial_wall.astype(np.float32)
atlas_cifti = atlas[(mask_medial_wall.flatten()) == 1]

def process_subject(subid, labels):
    path_timeseries = f'/media/afarahani/Expansion/Aging_HCP/{subid}_V1_MR/MNINonLinear/Results/'
    files = [
        'rfMRI_REST1_AP/rfMRI_REST1_AP_Atlas_MSMAll_hp0_clean.dtseries.nii',
        'rfMRI_REST1_PA/rfMRI_REST1_PA_Atlas_MSMAll_hp0_clean.dtseries.nii',
        'rfMRI_REST2_AP/rfMRI_REST2_AP_Atlas_MSMAll_hp0_clean.dtseries.nii',
        'rfMRI_REST2_PA/rfMRI_REST2_PA_Atlas_MSMAll_hp0_clean.dtseries.nii'
    ]

    data_list = []
    for file in files:
        inputFile = os.path.join(path_timeseries, file)
        if os.path.exists(inputFile):
            img = nib.cifti2.load(inputFile)
            data = img.get_fdata()[:, :globals.num_cort_vertices_noMW]
            # Demeaning
            data = data - np.mean(data, axis = 0)
            data_list.append(data)
        else:
            print(f"File {inputFile} not found, skipping.")

    if not data_list:
        print(f"No data found for subject {subid}, skipping subject.")
        return
    data_zc_list = []
    for data in data_list:
        data_zc = np.zeros((globals.nnodes_Schaefer, np.shape(data)[0]))
        for n in range(1, globals.nnodes_Schaefer + 1):
            # Parcel averaging
            data_zc[n - 1, :] = np.nanmean(data[:, atlas_cifti == n], axis = 1)

        # Z-Scoring
        data_zc = zscore(data_zc, axis = 1)
        data_zc_list.append(data_zc)

    # Concatenation of runs
    data = np.concatenate(data_zc_list, axis = 1)

    # Compute functional connectome and save it in path_FC
    FC = np.corrcoef(data)
    np.save(path_FC + 'FC_' + subid + '.npy', FC)

#------------------------------------------------------------------------------
# Process subjects
#------------------------------------------------------------------------------

c = 0
for s, subid in enumerate(subject_ids):
    print(s)
    path_timeseries = path_mri + 'Aging_HCP/' + subid + '_V1_MR/MNINonLinear/Results/'
    process_subject(subid, atlas)
    c += 1
    print('Processed subject number:', c)

#------------------------------------------------------------------------------
# END