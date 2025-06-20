"""

# Voxel-Wise Perfusion Similarity Between Men and Women

This script analyzes voxel-wise perfusion data for men and women across a set of subjects. It computes the correlation
of mean perfusion values across male and female subjects at the voxel level using both Spearman's rank correlation
coefficient and Pearson's correlation coefficient. The results are visualized in a scatter plot.

## Results:
    **Spearman's Rank Correlation (voxel-wise):**
    Spearman correlation coefficient: 0.985655505577404

    **Pearson's Correlation (voxel-wise):**
    Pearson correlation coefficient: 0.9884513576811998

# I performed a spin test with n = 1000, considering Pearson correlation.
* This was done after parcellation was applied (Schaefer-400 parcellation)

    r (parcel-wise): 0.9732319613351254
    p-spin: 0.000999000999000999

Note: Related to Fig.S2a,b.

"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import globals
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython import get_ipython
from scipy.stats import spearmanr, pearsonr
from functions import save_as_dscalar_and_npy
from functions import vasa_null_Schaefer, pval_cal
from globals import path_info_sub, path_results, path_figures
from functions import convert_cifti_to_parcellated_SchaeferTian

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------

warnings.filterwarnings("ignore")
get_ipython().magic('reset -sf')

#------------------------------------------------------------------------------
# Load subject information (development + aging)
#------------------------------------------------------------------------------

# Load aging subjects data
df = pd.read_csv(path_info_sub + 'clean_data_info.csv')
num_subjects = len(df)
ageold = np.array(df['interview_age'])/12
sexold = df.sex

# Load developmental subjects data
df = pd.read_csv(path_info_sub + 'Dev_clean_data_info.csv')
num_subjects = len(df)
agedev = np.array(df['interview_age'])/12
sexdev = df.sex

# Concatenate data from both cohorts
sex = np.concatenate((sexdev, sexold))
age = np.concatenate((agedev, ageold))

#------------------------------------------------------------------------------
# Load data (vertex-wise) - perfusion
#------------------------------------------------------------------------------

name = 'perfusion'

# Load aging cerebral blood perfusion data
old_data_vertexwise = np.load(path_results + name + '_all_vertex.npy')

# Load developmental cerebral blood perfusion data
dev_data_vertexwise = np.load(path_results + 'Dev_' + name + '_all_vertex.npy')

# Concatenate data across the cohorts
data_vertexwise = np.concatenate((dev_data_vertexwise, old_data_vertexwise), axis = 1)

#------------------------------------------------------------------------------
# Stratify data (perfusion map) based on biological sex of participants
#------------------------------------------------------------------------------

data_men = data_vertexwise[:, (sex == 'M')]
data_women = data_vertexwise[:, (sex == 'F')]

#------------------------------------------------------------------------------
# Plot men and women differences at vertex/voxel level
#------------------------------------------------------------------------------

cmap = plt.cm.get_cmap('turbo')
c1 = 'gray'
c2 = 'black'
plt.figure(figsize = (5, 5))
plt.scatter(np.mean(data_women, axis = 1)[:globals.num_cort_vertices_noMW],
            np.mean(data_men, axis = 1)[:globals.num_cort_vertices_noMW],
            color = c1,
            alpha = 0.7,
            s = 0.1)
plt.scatter(np.mean(data_women, axis = 1)[globals.num_cort_vertices_noMW:],
            np.mean(data_men, axis = 1)[globals.num_cort_vertices_noMW:],
            color =  c2,
            alpha = 0.7,
            s = 0.1)
plt.axline((np.min(np.mean(data_men, axis = 1)),
            np.min(np.mean(data_men, axis = 1))),
           (np.max(np.mean(data_women, axis = 1)),
            np.max(np.mean(data_women, axis = 1))),
           linewidth = 1,
           alpha = 0.7,
           color = 'black')
plt.xlim(10, 130)
plt.ylim(10, 130)
plt.gca().set_aspect('equal', adjustable = 'box')
plt.xticks(np.arange(10, 131, 40), labels = [])
plt.yticks(np.arange(10, 131, 40), labels = [])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(path_figures + name + '_scatterplot_men_women_voxelwise.png',
            format = 'png')
plt.show()

print('vertexwise spearmanr - 92k:')
print(spearmanr(np.mean(data_women, axis = 1), np.mean(data_men, axis = 1)))

print('vertexwise pearsonr - 92k:')
print(pearsonr(np.mean(data_women, axis = 1), np.mean(data_men, axis = 1)))

#------------------------------------------------------------------------------
# Save the mean perfusion map across men and women
#------------------------------------------------------------------------------

mean_women = np.mean(data_women, axis = 1)
mean_men = np.mean(data_men, axis = 1)

save_as_dscalar_and_npy(mean_women,
                        'cortex_subcortex',
                        path_results,
                        'mean_women')

save_as_dscalar_and_npy(mean_men,
                        'cortex_subcortex',
                        path_results,
                        'mean_men')

#------------------------------------------------------------------------------
# Perform spin test to create a p_value to report - parcellated data - Schaefer-400
#------------------------------------------------------------------------------

numspins = 1000

# if spin is not computed generate it - uncomment
nulls_index_vasa = vasa_null_Schaefer(numspins)
np.save(path_results + 'spin_men_women.npy', nulls_index_vasa)

nulls_index_vasa = np.load(path_results + 'spin_men_women.npy')

def corr_spin(x, y, spins, nspins):
    """
    Spin test - account for spatial autocorrelation
    """
    rho, _ = pearsonr(x, y)
    null = np.zeros((nspins,))

    # null correlation
    for i in range(nspins):
         null[i], _ = pearsonr(x, y[spins[:, i]])
    return rho, null

mean_women_parcels = convert_cifti_to_parcellated_SchaeferTian(mean_women.reshape(1, globals.num_vertices_voxels),
                                                        'cortex',
                                                        'X',
                                                         path_results,
                                                        'mean_women')

mean_men_parcels = convert_cifti_to_parcellated_SchaeferTian(mean_men.reshape(1, globals.num_vertices_voxels),
                                                        'cortex',
                                                        'X',
                                                         path_results,
                                                        'mean_men')

r, generated_null = corr_spin(mean_women_parcels.flatten(),
                             mean_men_parcels.flatten(),
                             nulls_index_vasa,
                             numspins)
p_value = pval_cal(r, generated_null, numspins)

print('r (parcel-wise): ' + str(r))
print('p-spin: ' + str(p_value))

#------------------------------------------------------------------------------
# END
