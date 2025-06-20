"""

Subcortical age coeff (linear effect) for HCP-A dataset.

Note: Related to Fig.S11b.

"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import globals
import warnings
import numpy as np
import matplotlib.pyplot as plt
from IPython import get_ipython
from globals import parcel_names_S4
from globals import path_results, path_figures
from functions import convert_cifti_to_parcellated_SchaeferTian

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------

warnings.filterwarnings("ignore")
get_ipython().magic('reset -sf')

#------------------------------------------------------------------------------
# Load age beta values for HCP-A
#------------------------------------------------------------------------------

map_data = np.load( path_results + 'Aging_betas_age_perfusion.npy')

#------------------------------------------------------------------------------
# Visualize beta values in subcortex
#------------------------------------------------------------------------------

data_parcelwise_subcortex = convert_cifti_to_parcellated_SchaeferTian(map_data.reshape(1, globals.num_vertices_voxels),
                                            'subcortex_double',
                                            'S4',
                                            path_results,
                                            'aging_beta_age_0')

data_parcelwise_subcortex_2save = np.zeros(globals.nnodes_S4)
data_parcelwise_subcortex_2save[:globals.nnodes_half_subcortex_S4] = data_parcelwise_subcortex[:, 0]
data_parcelwise_subcortex_2save[globals.nnodes_half_subcortex_S4:] = data_parcelwise_subcortex[:, 0]

#------------------------------------------------------------------------------
# How much beta we have per parcel?
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
# Sort the parcels by age-effect values and visualize them
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
plt.savefig(path_figures + 'age_beta_aging_subcortex.svg',
            format = 'svg')
plt.show()

#------------------------------------------------------------------------------
# END
