"""

Subcortical arterial territories

Note: Related to Fig.S14.

"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import os
import globals
import warnings
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from IPython import get_ipython
from globals import path_results, path_figures
from functions import convert_cifti_to_parcellated_SchaeferTian

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------

warnings.filterwarnings("ignore")
get_ipython().magic('reset -sf')

#------------------------------------------------------------------------------
# Gradient maps
#------------------------------------------------------------------------------

gradients = np.zeros((globals.nnodes_half_subcortex_S4, 2))

gradient_temp = os.path.join(path_results, 'gradient_0_perfusion.dscalar.nii')
gradient_temp_data = nib.cifti2.load(gradient_temp).get_data()
gradients[:, 0] = convert_cifti_to_parcellated_SchaeferTian(gradient_temp_data.reshape(1, globals.num_vertices_voxels),
                                          'subcortex_double',
                                          'S4',
                                          path_results,
                                          'G0').flatten()
gradient_temp = os.path.join(path_results, 'gradient_1_perfusion.dscalar.nii')
gradient_temp_data = nib.cifti2.load(gradient_temp).get_data()
gradients[:, 1] = (convert_cifti_to_parcellated_SchaeferTian(gradient_temp_data.reshape(1, globals.num_vertices_voxels),
                                          'subcortex_double',
                                          'S4',
                                          path_results,
                                          'G1').flatten())

#------------------------------------------------------------------------------
# Name of S4-Tian parcels
#------------------------------------------------------------------------------

a_list = []
with open(globals.parcel_names_S4) as f:
    a = f.readlines()
    pattern = r'([\n.]+)'
    for line in a:
        a_list.append(line)
parcel_names = np.array(a_list)[:27]

#------------------------------------------------------------------------------
# Sort subcortical parcels based on G1 values
#------------------------------------------------------------------------------

# Sort parcels based on G1 values (the second column in gradients)
sorted_indices = np.argsort(gradients[:, 1])  # Sort by G1
sorted_gradients = gradients[sorted_indices]  # Apply sorting
sorted_parcel_names = parcel_names[sorted_indices]  # Sort parcel names accordingly

#------------------------------------------------------------------------------
# Plot G0 and G1 in separate subplots
#------------------------------------------------------------------------------

# Create a figure with two subplots, one for G0 and one for G1
fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (12, 10), sharex = True)
x = np.arange(len(sorted_parcel_names))

# Plot G0 values in the first subplot
ax1.bar(x, sorted_gradients[:, 0], color= 'blue')
ax1.set_ylabel('G0 Gradient Value', fontsize = 14)
ax1.set_title('Subcortical Gradient Values (G0 and G1)', fontsize = 16)

# Plot G1 values in the second subplot
ax2.bar(x, sorted_gradients[:, 1], color = 'orange')
ax2.set_ylabel('G1 Gradient Value', fontsize = 14)
ax2.set_xlabel('Subcortical Parcels', fontsize = 14)

ax2.set_xticks(x)
ax2.set_xticklabels(sorted_parcel_names, rotation = 90)
plt.tight_layout()
plt.savefig(os.path.join(path_figures, 'subcortical_gradients_two_subplots.svg'),
            format = 'svg')
plt.show()

#------------------------------------------------------------------------------
# END