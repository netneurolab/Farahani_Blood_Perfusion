"""

Extract the perfusion score values for the S4 Tian parcellation.

Note: Related to Fig.S6.

"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import globals
import warnings
import numpy as np
from IPython import get_ipython
import matplotlib.pyplot as plt
from functions import convert_cifti_to_parcellated_SchaeferTian
from functions import save_parcellated_data_in_SchaeferTian_forVis
from globals import path_results, path_figures, parcel_names_S4

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------

warnings.filterwarnings("ignore")
get_ipython().magic('reset -sf')

#------------------------------------------------------------------------------
# Save main principal components of perfusion
#------------------------------------------------------------------------------

perfusion_mean_voxelwise = np.load(path_results + 'perfusion_PCscore.npy')[:, 0]
data_parcelwise_subcortex = convert_cifti_to_parcellated_SchaeferTian(perfusion_mean_voxelwise.reshape(1, globals.num_vertices_voxels),
                                            'subcortex_double',
                                            'S4',
                                            path_results,
                                            'perfusion_PCscore_0')

data_parcelwise_subcortex_2save = np.zeros(globals.nnodes_S4)
data_parcelwise_subcortex_2save[:globals.nnodes_half_subcortex_S4] = data_parcelwise_subcortex[:, 0]
data_parcelwise_subcortex_2save[globals.nnodes_half_subcortex_S4:] = data_parcelwise_subcortex[:, 0]

save_parcellated_data_in_SchaeferTian_forVis(data_parcelwise_subcortex_2save,
                                            'subcortex',
                                            'S4',
                                            path_results,
                                            'perfusion_PCscore_0_identicalLR')

#------------------------------------------------------------------------------
# How much perfusion (PC) do we have per parcel?
#------------------------------------------------------------------------------

# Get parcel names
a_list = []
with open(parcel_names_S4) as f:
    a = f.readlines()
    pattern = r'([\n.]+)'
    for line in a:
        a_list.append(line)
parcel_names = np.array(a_list)[:27]

#------------------------------------------------------------------------------
# Sort the parcels by mean PC values and visualize them
#------------------------------------------------------------------------------

sorted_indices_tot = np.argsort(data_parcelwise_subcortex.flatten())[::-1]
sorted_perfusion_tot = data_parcelwise_subcortex[sorted_indices_tot].flatten()
sorted_names_tot = parcel_names[sorted_indices_tot]

top_indices = np.argsort(sorted_perfusion_tot)
top_perfusion = sorted_perfusion_tot[top_indices]
top_names = np.array(sorted_names_tot)[top_indices]
plt.figure(figsize = (8, 5))
bars = plt.bar(range(27),
               top_perfusion,
               tick_label = top_names,
               color = 'gray')
plt.xticks(rotation = 90)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(path_figures + 'parcels_pc0_perfusion_subcortex.svg',
            format = 'svg')
plt.show()
#------------------------------------------------------------------------------
# END