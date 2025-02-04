"""

Neurosynth terms' maps - parcellated based on Schaefer-400 and TianS4 parcellation
This script processes Neurosynth terms by creating parcellated maps from raw NIfTI files using the Schaefer-Tian S4 atlas.

## Output Files:
    `neurosynth_data_S4.npy`: Contains parcellated Neurosynth data for both cortical and subcortical regions.
    `neurosynth_names.npy`: Contains the names of the Neurosynth terms.
    `.dscalar.nii` files for each Neurosynth term, back-projected onto the Schaefer-Tian S4 template for visualization purposes.

Note: In total, there exists 124 terms.

"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import os
import globals
import warnings
import numpy as np
import nibabel as nib
from IPython import get_ipython
from neuromaps import transforms
from functions import save_parcellated_data_in_SchaeferTian_forVis
from globals import path_neurosynth_raw, path_neurosynth, path_atlasV

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------

warnings.filterwarnings("ignore")
get_ipython().magic('reset -sf')

#------------------------------------------------------------------------------
# get neurosynth maps
#------------------------------------------------------------------------------

tian_load = nib.load(path_atlasV + 'Schaefer2018_400Parcels_7Networks_order_Tian_Subcortex_S4_MNI152NLin6Asym_1mm.nii.gz')
tian = transforms.mni152_to_mni152(tian_load,
                                    target = '1mm',
                                    method = 'nearest')
tian = tian.get_data()

# Retrieve all subfolder names within the raw Neurosynth data directory.
subfolder_names = []

# Go through each subfolder in the parent folder
for subfolder in os.listdir(path_neurosynth_raw):
    subfolder_path = os.path.join(path_neurosynth_raw, subfolder)
    if os.path.isdir(subfolder_path):
        subfolder_names.append(subfolder)

num_maps = len(subfolder_names)
parcel_out_maps = np.zeros((globals.nnodes_Schaefer_S4, num_maps))
parcellated = {}

r = 0 # Counter for the terms
for subfolder in os.listdir(path_neurosynth_raw):

    # Load a Neurosynth NIfTI file
    subfolder_path = os.path.join(path_neurosynth_raw, subfolder)
    nii_file_path = os.path.join(subfolder_path, "association-test_z.nii.gz")
    nii_data = nib.load(nii_file_path)
    print(str(nii_data.shape))

    # Transform data
    nib_transformed = transforms.mni152_to_mni152(nii_data,
                                                  target = '1mm')
    print(str(nib_transformed.shape))
    print('----------------')

    data = nib_transformed.get_data()
    if len(data.shape) == 4:
        data = data[:, :, :, 0]

    # Parcellate data
    for i in range(globals.nnodes_Schaefer_S4):
        parcel_out_maps[i, r] = np.nanmean(data[tian == 1 + i])

    # Save the parcellated data for visualization
    save_parcellated_data_in_SchaeferTian_forVis(parcel_out_maps[:,r].T,
                                                 'cortex_subcortex',
                                                 'S4',
                                                 path_neurosynth,
                                                 subfolder_names[r])
    r = r + 1

np.save(path_neurosynth + 'neurosynth_data_S4.npy', parcel_out_maps)
np.save(path_neurosynth + 'neurosynth_names.npy', subfolder_names)

#------------------------------------------------------------------------------
# END