"""

# Assess the correspondence between CBF (First Principal Component) and FC Strength PC across participants

r: 0.26344339352511725
p_perm: 0.000999000999000999

n_permutations = 1000

Note: Related to Fig.S4c.

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
import matplotlib.colors as mcolors
from globals import path_FC, path_figures
from scipy.stats import linregress, pearsonr
from globals import path_results, path_info_sub
from functions import convert_cifti_to_parcellated_SchaeferTian

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------

warnings.filterwarnings("ignore")
get_ipython().magic('reset -sf')
nnodes = 400

#------------------------------------------------------------------------------
# Load subject information
#------------------------------------------------------------------------------

df = pd.read_csv(path_info_sub + 'clean_data_info.csv')
num_subjects = len(df)
age_old = df.interview_age
sex_old = df.sex
subject_ids_age = df['src_subject_id']

df = pd.read_csv(path_info_sub + 'Dev_clean_data_info.csv')
num_subjects = len(df)
age_dev = df.interview_age
sex_dev = df.sex
subject_ids_dev = df['src_subject_id']

# Merge data across development and aging
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

# Combine Development + Aging FC data
FCs_all = np.concatenate((FCs_dev, FCs_age), axis = 0)
num_sub = len(FCs_all)

# Compute node-wise degree (strength) from the FC matrix
FCs = np.abs(FCs_all)
fc_deg = np.zeros((num_sub, globals.nnodes_Schaefer))
for s in range(num_sub):
    fc_deg[s, :] = degree.strengths_und(abs(FCs[s, :, :]))
    print(s)

# Compute overall FC strength by averaging across all nodes for each subject.
fc_deg_unique = np.mean(fc_deg, axis = 1)

#------------------------------------------------------------------------------
# Load Cerebral Blood Perfusion (CBF) Data
#------------------------------------------------------------------------------

name = 'perfusion'
old_data_vertexwise = np.load(path_results + name + '_all_vertex.npy')
dev_data_vertexwise = np.load(path_results + 'Dev_' + name + '_all_vertex.npy')
data_vertexwise = np.concatenate((dev_data_vertexwise, old_data_vertexwise), axis = 1)

perfusion_parcel = convert_cifti_to_parcellated_SchaeferTian(data_vertexwise.T,
                                          'cortex',
                                          'X',
                                          path_results,
                                          'data_vertexwise')

# Compute an overall perfusion value across all parcels for each subject
perfusion_parcelwise_unique = np.mean(perfusion_parcel, axis = 0)

#------------------------------------------------------------------------------
# Are these related? (perfusion and FC deg) - individual-level
#------------------------------------------------------------------------------

fc_deg_unique = fc_deg_unique/400
# Compute Pearson correlation
correlation, p_value = pearsonr(fc_deg_unique, perfusion_parcelwise_unique)
print(f"Pearson correlation: {correlation}, P-value: {p_value}")

# Linear regression for best-fit line
slope, intercept, _, _, _ = linregress(fc_deg_unique, perfusion_parcelwise_unique)
x_line = np.linspace(fc_deg_unique.min(), fc_deg_unique.max(), 100)
y_line = slope * x_line + intercept

# Scatter plot
plt.figure(figsize = (5, 5))
cmap = plt.get_cmap('Greys')
norm = mcolors.Normalize(vmin = age.min(),
                         vmax = age.max())
colors = cmap(norm(age)) # Map each age value to a color
scatter = plt.scatter(fc_deg_unique, perfusion_parcelwise_unique,
                      facecolor = colors,
                      edgecolor = 'black',
                      s = 50,
                      alpha = 0.6)
plt.plot(x_line, y_line,
         color = 'black',
         linewidth = 2,
         label ='Best-Fit Line')
cbar = plt.colorbar(mappable = plt.cm.ScalarMappable(norm = norm, cmap = cmap),
                    ax = plt.gca())
cbar.set_label('Age (years)')
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({'font.size': 8})
plt.xlabel('FC Degree')
plt.ylabel('Perfusion')
plt.legend()
plt.tight_layout()
plt.savefig(path_figures + 'FC_blood_individual_subjects_real_age_colored_with_line.svg',
            format = 'svg')
plt.show()

#------------------------------------------------------------------------------
# Permutation testing for correlation significance
#------------------------------------------------------------------------------

# Observed correlation
observed_corr, _ = pearsonr(fc_deg_unique, perfusion_parcelwise_unique)
print(f"Observed correlation: {observed_corr}")

# Number of permutations
n_permutations = 1000
permuted_corrs = np.zeros(n_permutations)

# Generate permuted correlations
for i in range(n_permutations):
    shuffled_fc_deg = np.random.permutation(fc_deg_unique) # Shuffle FC degree
    permuted_corrs[i], _ = pearsonr(shuffled_fc_deg, perfusion_parcelwise_unique)

# Calculate p-value as proportion of permuted correlations ≥ observed
p_value = (np.sum(np.abs(permuted_corrs) >= np.abs(observed_corr)) + 1) / (n_permutations + 1)
print(f"P-value from permutation test: {p_value}")

#------------------------------------------------------------------------------
# Plot the null distribution of permuted correlations
#------------------------------------------------------------------------------

plt.figure(figsize = (8, 5))
plt.hist(permuted_corrs,
         bins = 50,
         alpha = 0.75,
         color = 'gray',
         edgecolor = 'black')
plt.axvline(x = observed_corr,
            color = 'red',
            linestyle = '--',
            label = f'Observed corr = {observed_corr:.3f}')
plt.axvline(x = -observed_corr,
            color = 'red',
            linestyle = '--')
plt.title('Permutation Test: Null Distribution of Correlations')
plt.xlabel('Correlation Coefficient')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.savefig(path_figures + 'permutation_test_null_distribution.svg',
            format = 'svg')
plt.show()

#------------------------------------------------------------------------------
# END
