"""

# Assess the correspondence between the first principal component (PC) of CBF 
  and the first PC of FC strength.

r      = 0.5190785302691128
p_spin = 0.002997002997002997

Note: Related to Fig.1c.

"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import globals
import warnings
import numpy as np
import pandas as pd
from bct import degree
from pathlib import Path
import matplotlib.pyplot as plt
from IPython import get_ipython
from sklearn.decomposition import PCA
from scipy.stats import zscore, pearsonr
from numpy.polynomial.polynomial import Polynomial
from functions import convert_cifti_to_parcellated_SchaeferTian
from globals import path_FC, path_info_sub, path_results, path_figures
from functions import save_parcellated_data_in_SchaeferTian_forVis, pval_cal

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------

warnings.filterwarnings("ignore")
get_ipython().magic('reset -sf')

#------------------------------------------------------------------------------
# Needed Functions
#------------------------------------------------------------------------------

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

#------------------------------------------------------------------------------
# Load subject information
#------------------------------------------------------------------------------

df = pd.read_csv(path_info_sub + 'clean_data_info.csv')
num_subjects = len(df)
age_old = df.interview_age
sex_old = df.sex
subject_ids_aging = df['src_subject_id']

df = pd.read_csv(path_info_sub + 'Dev_clean_data_info.csv')
num_subjects = len(df)
age_dev = df.interview_age
sex_dev = df.sex
subject_ids_dev = df['src_subject_id']

# Merge across development & aging
sex = np.concatenate((sex_dev, sex_old))
age = np.concatenate((age_dev, age_old))

#------------------------------------------------------------------------------
# Load FC Matrices and calculate their strength
#------------------------------------------------------------------------------

# Development
FC_all = []
index_to_save = []
File_exists = []
for s, subid in enumerate(subject_ids_dev):
    my_file = Path(path_FC + 'Dev_FC_' + subid + '.npy')
    if my_file.is_file():
        FC_all.append(np.load(path_FC + 'Dev_FC_' + subid + '.npy'))
        File_exists.append(subid)
        index_to_save.append(s)

FCs_dev = np.array(FC_all)

# Aging
FC_all = []
index_to_save = []
File_exists = []
for s, subid in enumerate(subject_ids_aging):
    my_file = Path(path_FC + 'FC_' + subid + '.npy')
    if my_file.is_file():
        FC_all.append(np.load(path_FC + 'FC_' + subid + '.npy'))
        File_exists.append(subid)
        index_to_save.append(s)
FCs_age = np.array(FC_all)

# Concatenate: Developmental + Aging
FCs_all = np.concatenate((FCs_dev, FCs_age), axis = 0)

# Compute the node-wise degree (strength) from FC.
FCs = np.abs(FCs_all)
fc_deg = np.zeros((len(FCs_all), globals.nnodes_Schaefer))
for s in range(len(FCs_all)):
    fc_deg[s, :] = degree.strengths_und(abs(FCs[s, :, :]))
    print(s)

#------------------------------------------------------------------------------
# Load the first principal component score map of cerebral blood perfusion
#------------------------------------------------------------------------------

scores_data = np.load(path_results + 'perfusion_PCscore.npy')[:, 0]
perfusion_pc = convert_cifti_to_parcellated_SchaeferTian(scores_data.reshape(1, globals.num_vertices_voxels),
                                                      'cortex',
                                                      'S1',
                                                      path_results,
                                                      'test')

#------------------------------------------------------------------------------
# Assess the co-localization of the perfusion score map with the FC strength score map
#------------------------------------------------------------------------------

# Spin Permutation Setup
nspins = 1000
'''
nulls_index_vasa = vasa_null_Schaefer(nspins)
np.save(path_results + 'spin_metabolism.npy', nulls_index_vasa)
'''
nulls_index_vasa = np.load(path_results + 'spin_metabolism.npy')
spin_res = []
for i in range(nspins):
    spin_res.append(perfusion_pc[nulls_index_vasa[:, i], 0])

# Z-score node-wise FC strengths across subjects
fc_deg_zscore = zscore(fc_deg, axis = 1)

# PCA on parcellated FC strengths
pca = PCA(n_components = 1)
fc_deg_pc = pca.fit_transform(fc_deg_zscore.T)

# Flip the sign for convenience
fc_deg_pc = -1 * fc_deg_pc

# Save PC for visualization if desired
save_parcellated_data_in_SchaeferTian_forVis(fc_deg_pc,
                        'cortex',
                        'X',
                        path_results,
                        'PC_fc_deg')

# Calculate Correlation + Spin-Test
r, generated_null = corr_spin(fc_deg_pc.flatten(),
                              perfusion_pc.flatten(),
                              nulls_index_vasa,
                              nspins)
p_value = pval_cal(r, generated_null, nspins)

print('correlation of FC degree and perfusion - parcelwise: ')
print(pearsonr(fc_deg_pc.flatten(), perfusion_pc))
print('the p-value is: ')
print(p_value)

#------------------------------------------------------------------------------
# Visualization - group level (PCs)
#------------------------------------------------------------------------------

plt.figure(figsize = (5, 5))
plt.scatter(fc_deg_pc, perfusion_pc,
            s = 15,
            color = 'gray')
p = Polynomial.fit(
    fc_deg_pc.flatten(),
    perfusion_pc.flatten(),
    1)
plt.plot(*p.linspace(),
         color = 'black',
         linewidth = 1)
plt.ylim(-56, 66)
plt.title('FC-score and CBF score maps')
plt.yticks(np.arange(-56, 66, 10))
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({'font.size': 8})
plt.tight_layout()
plt.savefig(path_figures + 'scatter_PCperfusion_PCFCDeg.svg',
            format = 'svg')
plt.show()
#------------------------------------------------------------------------------
# END
