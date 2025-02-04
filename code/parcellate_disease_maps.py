"""
-------------------------------------------------------------------------------

Disease Map Voxel-Based Morphometry (VBM) Data Processing

This script processes disease map VBM data as described in the following study:

Patterns of Atrophy in Pathologically Confirmed Dementias: A Voxelwise Analysis
Related Article: http://doi.org/10.1136/jnnp-2016-314978

Data Included:
- 186 patients diagnosed with dementia, categorized into:
    - 107 Alzheimer’s Disease (AD):
        - 68 Early-Onset AD (EOAD)
        - 29 Late-Onset AD (LOAD)
        - 10 Presenilin-1 Mutation Carriers (PS-1)
    - 25 Dementia with Lewy Bodies (DLB)
    - 11 3-Repeat Tauopathy (3Rtau)
    - 17 4-Repeat Tauopathy (4Rtau)
    - 12 TDP43A (FTLD-TDP, type A)
    - 14 TDP43C (FTLD-TDP, type C)
- 73 Healthy Controls

-------------------------------------------------------------------------------

Processing Steps:
1. **Find Schaefer-400 Parcellation in TPM Space**:
    The disease maps are defined in TPM space. 
    To align them with the Schaefer-400 parcellation, perform the following registration using FSL tools.

    **FSL Commands**:

    # Linear registration using FLIRT
    flirt -in ./template_VBM/TPM.nii 
    -ref ./mni_icbm152_nlin_sym_09c/mni_icbm152_t1_tal_nlin_sym_09c.nii
    -out ./data_2009cSym_MNI152/TPM_to_MNI_linear
    -omat ./data_2009cSym_MNI152/TPM_to_MNI_linear.mat
    -bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12

    # Non-linear registration using FNIRT
    fnirt --in=./template_VBM/TPM.nii
    --aff=./data_2009cSym_MNI152/TPM_to_MNI_linear.mat
    --cout=./data_2009cSym_MNI152/TPM_to_MNI_nonlinear_coeff
    --ref=./mni_icbm152_nlin_sym_09c/mni_icbm152_t1_tal_nlin_sym_09c.nii
    --iout=./data_2009cSym_MNI152/TPM_to_MNI_nonlinear

    # Invert warp using INVWARP
    invwarp --ref=./template_VBM/TPM.nii
    --warp=./data_2009cSym_MNI152/TPM_to_MNI_nonlinear_coeff
    --out=./data_2009cSym_MNI152/MNI_to_TPM_nonlinear_coeff

    # Apply warp to Schaefer-400 atlas
    applywarp --ref=./template_VBM/TPM.nii
    --in=./atlas/schaefer400.nii
    --warp=./atlas/schaefer400.nii
    --warp=./data_2009cSym_MNI152/MNI_to_TPM_nonlinear_coeff
    --out=/atlas_disease/schaefer400_TPM --interp=nn

**Note**: The resulting `schaefer400_TPM.nii.gz` file is utilized in this script.

**Required Files**:
- **TPM.nii**: https://github.com/spm/spm12/blob/3085dac00ac804adb190a7e82c6ef11866c8af02/tpm/TPM.nii
  - Resolution: 1.5mm
  - Dimensions: 121x145x121

2. **Parcellate the Group-Average Maps in TPM Space**:
    - The following Python script performs the parcellation of the registered disease maps using the Schaefer-400 atlas.

-------------------------------------------------------------------------------

"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import os
import warnings
import numpy as np
import globals
from IPython import get_ipython
from neuromaps.images import load_data
from neuromaps.images import dlabel_to_gifti
from netneurotools import datasets as nntdata
from globals import path_atlas_disease, path_disease, path_disease_raw
from functions import load_nifti, save_parcellated_data_in_SchaeferTian_forVis

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------

warnings.filterwarnings("ignore")
get_ipython().magic('reset -sf')

#------------------------------------------------------------------------------
# Needed functions
#------------------------------------------------------------------------------

def calculate_mean_w_score(w_score, atlas, n_parcels):
    """
    Calculate the mean w-score for each parcel in the atlas.
    Returns:
    -------
    mean_w_score :
        Array containing the mean w-score for each parcel.
    """
    mean_w_score = np.zeros((int(n_parcels), 1))
    for i in range(1, int(n_parcels) + 1):
        # Create a mask for the current parcel
        parcel_mask = (atlas == i)

        # Calculate the mean, excluding NaNs
        if np.any(parcel_mask):  # Check if there are any voxels in the parcel
            mean_w_score[i - 1, :] = np.nanmean(w_score[parcel_mask])
        else:
            mean_w_score[i - 1, :] = np.nan # Assign NaN if no voxels meet the criteria
    return mean_w_score

#------------------------------------------------------------------------------
# Disease names
#------------------------------------------------------------------------------

disease = ['PS1',
           'EOAD',
           'LOAD',
           '3Rtau',
           '4Rtau',
           'TDP43C',
           'TDP43A',
           'DLB']

#------------------------------------------------------------------------------
# Load schaefer atlas
#------------------------------------------------------------------------------

nnodes  =  globals.nnodes_Schaefer # nnodes = 400
schafer_atlas = load_nifti(os.path.join(path_atlas_disease, f'schaefer{nnodes}_TPM.nii.gz'))

#------------------------------------------------------------------------------
# Parcellate each disease map
#------------------------------------------------------------------------------

for d in disease:
    # Load group-averaged disease map
    w_score_mean = -1 * load_nifti(path_disease_raw + d + '.nii.gz')

    # Calculate mean w-scores for Schaefer parcels - parcellation step
    w_score_schaefer  = calculate_mean_w_score(w_score_mean, schafer_atlas, nnodes)

    # Save the parcellated results to .npy format
    np.save(path_disease + 'mean_atropy_' + d + '_Schaefer.npy',
            w_score_schaefer)

    # Visualize and save cortical disease maps
    schaefer = nntdata.fetch_schaefer2018('fslr32k')[str(nnodes) + 'Parcels7Networks']
    atlas = load_data(dlabel_to_gifti(schaefer))
    file_name = 'mean_atropy_' + d + '_Schaefer'
    save_parcellated_data_in_SchaeferTian_forVis(w_score_schaefer,
                                                 'cortex',
                                                 'X',
                                                 path_disease,
                                                 file_name)

#------------------------------------------------------------------------------
# END