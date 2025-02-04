"""

Calculate the Spearman correlation between age and perfusion for each biological sex group
 stratified by dataset (HCP-D and HCP-A)

The Spearman correlation values are saved as follows:

    For the aging HCP:
    spearmanr_age_with_old_data_men.npy & spearmanr_age_with_old_data_men.dscalar.nii
    spearmanr_age_with_old_data_women.npy and spearmanr_age_with_old_data_women.dscalar.nii

    For the developmental HCP:
    spearmanr_age_with_dev_data_men.npy and spearmanr_age_with_dev_data_men.dscalar.nii
    spearmanr_age_with_dev_data_women.npy and spearmanr_age_with_dev_data_women.dscalar.nii

Note: Related to Fig.S8 & Fig.S11.

"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import globals
import warnings
import numpy as np
import pandas as pd
from IPython import get_ipython
from  scipy.stats  import spearmanr
from functions import save_as_dscalar_and_npy
from globals import path_info_sub, path_results

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------

warnings.filterwarnings("ignore")
get_ipython().magic('reset -sf')

#------------------------------------------------------------------------------
# Load subject information (development and aging)
#------------------------------------------------------------------------------

df = pd.read_csv(path_info_sub + 'clean_data_info.csv')
num_subjects = len(df)
ageold = np.array(df['interview_age'])/12
sexold = df.sex

df = pd.read_csv(path_info_sub + 'Dev_clean_data_info.csv')
num_subjects = len(df)
agedev = np.array(df['interview_age'])/12
sexdev = df.sex

#------------------------------------------------------------------------------
# Load data (vertex-wise) - perfusion (development and aging)
#------------------------------------------------------------------------------

old_data_vertexwise = np.load(path_results + 'perfusion_all_vertex.npy')
dev_data_vertexwise = np.load(path_results + 'Dev_perfusion_all_vertex.npy')

# Based on biological sex
old_data_men = old_data_vertexwise[:, sexold == 'M']
old_data_women = old_data_vertexwise[:, sexold == 'F']

dev_data_men = dev_data_vertexwise[:, sexdev == 'M']
dev_data_women = dev_data_vertexwise[:, sexdev == 'F']

#------------------------------------------------------------------------------
# Calculate Spearman's r of age with perfusion maps- stratified by sex and dataset
#------------------------------------------------------------------------------

old_men_r = np.zeros((globals.num_vertices_voxels, 1))
old_women_r = np.zeros((globals.num_vertices_voxels, 1))
dev_men_r = np.zeros((globals.num_vertices_voxels, 1))
dev_women_r  = np.zeros((globals.num_vertices_voxels, 1))

for i in range(globals.num_vertices_voxels):
    old_men_r[i] = spearmanr(ageold[sexold == 'M'], old_data_men[i, :])[0]
    old_women_r[i] = spearmanr(ageold[sexold == 'F'], old_data_women[i, :])[0]
    dev_men_r[i] = spearmanr(agedev[sexdev == 'M'], dev_data_men[i, :])[0]
    dev_women_r[i]  = spearmanr(agedev[sexdev == 'F'], dev_data_women[i, :])[0]
    print(i)

save_as_dscalar_and_npy(old_men_r,
                        'cortex_subcortex',
                        path_results,
                        'spearmanr_age_with_old_data_men')
save_as_dscalar_and_npy(old_women_r,
                        'cortex_subcortex',
                        path_results,
                        'spearmanr_age_with_old_data_women')

save_as_dscalar_and_npy(dev_men_r,
                        'cortex_subcortex',
                        path_results,
                        'spearmanr_age_with_dev_data_men')
save_as_dscalar_and_npy(dev_women_r,
                        'cortex_subcortex',
                        path_results,
                        'spearmanr_age_with_dev_data_women')

#------------------------------------------------------------------------------
# END