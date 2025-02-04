"""
# Sort the cortical areas with high perfusion based on glasser parcellation.

Note: Related to Table.S1.

"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import globals
import warnings
import numpy as np
import pandas as pd
from IPython import get_ipython
import matplotlib.pyplot as plt
from functions import convert_cifti_to_parcellated_Glasser
from globals import path_results, path_glasser

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------

warnings.filterwarnings("ignore")
get_ipython().magic('reset -sf')

perfusion_mean_voxelwise = np.load(path_results + 'perfusion_PCscore.npy')[: globals.num_cort_vertices_noMW, 0]

#------------------------------------------------------------------------------
# How much perfusion do we have per parcel of Glasser parcellation?
#------------------------------------------------------------------------------

# Hemisphere 1
perfusion_L = np.zeros((globals.nnodes_Glasser_half, 1))
perfusion_L = convert_cifti_to_parcellated_Glasser(perfusion_mean_voxelwise.reshape(1, globals.num_cort_vertices_noMW),
                                                   'l')
# Read the Excel file
df = pd.read_excel(path_glasser + 'Glasser_2016_Table.xlsx')
name_parcels = np.array(df['Area\nName'])[:globals.nnodes_Glasser_half]

# Sort the parcels by perfusion values
sorted_indices = np.argsort(perfusion_L.flatten())[::-1]
sorted_perfusion = perfusion_L[sorted_indices].flatten()
sorted_names = name_parcels[sorted_indices]

# Create the bar plot
plt.figure(figsize = (20, 15))
plt.bar(range(globals.nnodes_Glasser_half),
        sorted_perfusion,
        tick_label = sorted_names)
plt.xticks(rotation = 90)
plt.xlabel('right hem parcels')
plt.ylabel('Mean Perfusion')
plt.title('Mean Perfusion per Parcel (Sorted)')
plt.tight_layout()
plt.show()

# Hemisphere 2
perfusion_R = np.zeros((globals.nnodes_Glasser_half, 1))
perfusion_R = convert_cifti_to_parcellated_Glasser(perfusion_mean_voxelwise.reshape(1, globals.num_cort_vertices_noMW),
                                                   'r')
# Sort the parcels by perfusion values
sorted_indices = np.argsort(perfusion_R.flatten())[::-1]
sorted_perfusion = perfusion_R[sorted_indices].flatten()
sorted_names = name_parcels[sorted_indices]

# Create the bar plot
plt.figure(figsize = (20, 15))
plt.bar(range(globals.nnodes_Glasser_half),
        sorted_perfusion,
        tick_label = sorted_names)
plt.xticks(rotation = 90)
plt.xlabel('right hem parcels')
plt.ylabel('Mean Perfusion')
plt.title('Mean Perfusion per Parcel (Sorted)')
plt.tight_layout()
plt.show()

#------------------------------------------------------------------------------
# both hemispheres

perfusion_LR = np.zeros((globals.nnodes_Glasser_half, 1))
perfusion_LR = convert_cifti_to_parcellated_Glasser(perfusion_mean_voxelwise.reshape(1, globals.num_cort_vertices_noMW),
                                                    'lr_double')
perfusion_LR_vis = np.zeros((globals.nnodes_Glasser, 1))
perfusion_LR_vis[:globals.nnodes_Glasser_half] = perfusion_LR
perfusion_LR_vis[globals.nnodes_Glasser_half:] = perfusion_LR

# Sort the parcels by perfusion values
sorted_indices_tot = np.argsort(perfusion_LR.flatten())[::-1]
sorted_perfusion_tot = perfusion_LR[sorted_indices_tot].flatten()
sorted_names_tot = name_parcels[sorted_indices_tot]

# Get the top ten parcels
top_indices = np.argsort(sorted_perfusion_tot)
top_perfusion = sorted_perfusion_tot[top_indices]
top_names = np.array(sorted_names_tot)[top_indices]
'''
plt.figure(figsize = (15, 5))
bars = plt.bar(range(50),
               top_perfusion,
               tick_label = top_names,
               color = 'gray')
plt.xticks(rotation = 90)
plt.ylim(0, 60)
plt.yticks(np.arange(0, 60, 10))
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(path_figures +' top_parcels_pc0_perfusion.svg',
            format = 'svg')
plt.show()
'''
#------------------------------------------------------------------------------
data = { 'area_names' : top_names,
        'area_scores' : top_perfusion
        
        }
df = pd.DataFrame(data) # use this to create the table
#------------------------------------------------------------------------------
# END
