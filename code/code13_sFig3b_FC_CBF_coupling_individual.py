"""
# Assess the similarity between the FC strength map and the blood perfusion map for each participant.
The decoupling of the two occurs in the aging cohort.

    developmental - coupling correlation:
    rho = 0.0855493779658146
    p_perm = 0.025974025974025976

    aging - coupling correlation:
    rho =-0.23893399681473426
    p_perm = 0.000999000999000999

    n_permutations = 1000

Note: Related to Fig.S3b.

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
from IPython import get_ipython
import matplotlib.pyplot as plt
from globals import path_FC, path_figures
from scipy.stats import spearmanr, pearsonr
from globals import path_results, path_info_sub
from functions import convert_cifti_to_parcellated_SchaeferTian, pval_cal

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------

warnings.filterwarnings("ignore")
get_ipython().magic('reset -sf')
nnodes = 400

#------------------------------------------------------------------------------
# Load subject information
#------------------------------------------------------------------------------

# Load subject information
df = pd.read_csv(path_info_sub + 'clean_data_info.csv')
num_subjects = len(df)
age_old = df.interview_age/12
sex_old = df.sex
subject_ids_age = df['src_subject_id']

df = pd.read_csv(path_info_sub + 'Dev_clean_data_info.csv')
num_subjects = len(df)
age_dev = df.interview_age/12
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
AFC_all = []
Aindex_to_save = []
AFile_exists = []
for s, subid in enumerate(subject_ids_age):
    my_file = Path(path_FC + 'FC_' + subid + '.npy')
    if my_file.is_file():
        AFC_all.append(np.load(path_FC + 'FC_' + subid + '.npy'))
        AFile_exists.append(subid)
        Aindex_to_save.append(s)
FCs_age = np.array(AFC_all)

# Concatenate: Developmental + Aging
FCs_all = np.concatenate((FCs_dev, FCs_age), axis = 0)
num_sub = len(FCs_all)
FCs = np.abs(FCs_all)

# Compute node-wise degree (strength) from FC
fc_deg = np.zeros((num_sub, globals.nnodes_Schaefer))
for s in range(num_sub):
    fc_deg[s, :] = degree.strengths_und(abs(FCs[s, :, :]))
    print(s)

#------------------------------------------------------------------------------
# Load cerebral blood perfusion data
#------------------------------------------------------------------------------

name = 'perfusion'
old_data_vertexwise = np.load(path_results + name + '_all_vertex.npy')
dev_data_vertexwise = np.load(path_results + 'Dev_' + name + '_all_vertex.npy')

data_vertexwise = np.concatenate((dev_data_vertexwise, old_data_vertexwise), axis = 1)
perfusion_parcelwise = convert_cifti_to_parcellated_SchaeferTian(data_vertexwise.T,
                                          'cortex',
                                          'X',
                                          path_results,
                                          'perfusion_subset')

# -----------------------------------------------------------------------------
# Are these related? (perfusion and FC deg) - individual-level
# -----------------------------------------------------------------------------

corr_vals = np.zeros((num_sub, 1))

for n in range(num_sub):
    corr_vals[n, 0] = pearsonr(fc_deg[n, :].flatten(),
                               perfusion_parcelwise[:, n].flatten())[0]

plt.figure(figsize = (10, 5))
cmap = plt.get_cmap('coolwarm')
colors = cmap(np.linspace(0, 1, 100))
colors = [colors[90]  if s == 'F' else colors[10] for s in sex]

plt.scatter(age, corr_vals,
            color = colors,
            s = 15,
            alpha = 0.5)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({'font.size': 8})
plt.tight_layout()
plt.savefig(path_figures + 'FC_blood_individual_subjects.svg',
            format = 'svg')
plt.show()

#------------------------------------------------------------------------------
# Assess the FC strength and perfusion decoupling/coupling in aging and development
#------------------------------------------------------------------------------

corr_vals_dev = corr_vals[:len(sex_dev)]

print('developmental - coupling correlation:')
print(spearmanr(corr_vals_dev, age_dev))

corr_vals_aging =  corr_vals[len(sex_dev):]
print('aging - coupling correlation:')
print(spearmanr(corr_vals_aging, age_old))

#------------------------------------------------------------------------------
# Perform permutation testing to assess the significance of coupling/decoupling.
#------------------------------------------------------------------------------

n_permutations = 1000
corr_vals_dev = corr_vals[:len(sex_dev)]
corr_vals_aging = corr_vals[len(sex_dev):]

# Observed correlations
obs_corr_dev, _ = spearmanr(corr_vals_dev, age_dev)
obs_corr_aging, _ = spearmanr(corr_vals_aging, age_old)

# Permutation testing
perm_corr_dev = np.zeros(n_permutations)
perm_corr_aging = np.zeros(n_permutations)

for i in range(n_permutations):
    permuted_ages_dev = np.random.permutation(age_dev)
    perm_corr_dev[i], _ = spearmanr(corr_vals_dev, permuted_ages_dev)

    permuted_ages_aging = np.random.permutation(age_old)
    perm_corr_aging[i], _ = spearmanr(corr_vals_aging, permuted_ages_aging)

# Calculate p-values
p_value_dev = pval_cal(obs_corr_dev, perm_corr_dev, n_permutations)
p_value_aging = pval_cal(obs_corr_aging, perm_corr_aging, n_permutations)

#------------------------------------------------------------------------------
# END