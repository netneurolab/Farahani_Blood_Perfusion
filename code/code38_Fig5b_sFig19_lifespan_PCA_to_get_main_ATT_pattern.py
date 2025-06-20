"""

# Get the main pattern of arterial transit time (ATT) across participants.

The first PC gives the pattern of watershed regions.

We first z-score data of individual subjects and then apply PCA on the data.

Results of the PCA:

------------------------------------------

pca.explained_variance_ratio_

         0.22917116
         0.04073514
         0.01043232
         0.00492412
         0.00458851

------------------------------------------

Get the correlation of loadings with age: Lifespan
0
SpearmanrResult(correlation=0.19960438035154318, pvalue=3.410619932178759e-13)
--------------
1
SpearmanrResult(correlation=-0.7451320023780245, pvalue=1.727562510560648e-231)
--------------
2
SpearmanrResult(correlation=0.19937856058729156, pvalue=3.629583367476866e-13)
--------------
3
SpearmanrResult(correlation=-0.0596257372654084, pvalue=0.03125435271354315)
--------------
4
SpearmanrResult(correlation=0.07278915455175027, pvalue=0.008526876950011544)

------------------------------------------

Get the correlation of loadings with age: Development group

PC 0: Spearman correlation = -0.059633550302934525, p-value = 0.13581186510500232
--------------
PC 1: Spearman correlation = -0.3883460373482636, p-value = 5.30139930670677e-24
--------------
PC 2: Spearman correlation = 0.13724778282791708, p-value = 0.000568648671153376
--------------
PC 3: Spearman correlation = -0.026552041422133965, p-value = 0.5069126970460315
--------------
PC 4: Spearman correlation = -0.0059225619855368845, p-value = 0.8823379273556431

------------------------------------------

Get the correlation of loadings with age: Aging group

PC 0: Spearman correlation = 0.305442098588254, p-value = 4.1628589834822245e-16
--------------
PC 1: Spearman correlation = -0.5318660610043768, p-value = 8.894603271102867e-51
--------------
PC 2: Spearman correlation = 0.1382856656210174, p-value = 0.0003044748181938347
--------------
PC 3: Spearman correlation = 0.15070831715434588, p-value = 8.165514338056632e-05
--------------
PC 4: Spearman correlation = -0.11254558819599511, p-value = 0.003341595295158002
------------------------------------------

Note: Related to Fig.5b & Fig.S19.

"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython import get_ipython
from sklearn.decomposition import PCA
from scipy.stats import zscore, spearmanr
from functions import save_as_dscalar_and_npy
from globals import path_results, path_figures, path_info_sub

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------

warnings.filterwarnings("ignore")
get_ipython().magic('reset -sf')

#------------------------------------------------------------------------------
# Load subject information - demographics (development + aging)
#------------------------------------------------------------------------------

df = pd.read_csv(path_info_sub + 'clean_data_info.csv')
num_subjects = len(df)
ageold = np.array(df['interview_age'])/12
sexold = df.sex

df = pd.read_csv(path_info_sub + 'Dev_clean_data_info.csv')
num_subjects = len(df)
agedev = np.array(df['interview_age'])/12
sexdev = df.sex

sex = np.concatenate((sexdev, sexold))
age = np.concatenate((agedev, ageold))

#------------------------------------------------------------------------------
# Load data (vertex-wise) - perfusion in this case
#------------------------------------------------------------------------------

name = 'arrival'

old_data_vertexwise = np.load(path_results + name + '_all_vertex.npy')
dev_data_vertexwise = np.load(path_results + 'Dev_' + name + '_all_vertex.npy')
data_vertexwise = np.concatenate((dev_data_vertexwise, old_data_vertexwise), axis = 1)

#------------------------------------------------------------------------------
# Perform PCA - vertex wise - cortex and subcortex
#------------------------------------------------------------------------------

data_vertexwise_pca = zscore(data_vertexwise, axis = 0)
num_components = 5
pca = PCA(n_components = num_components, random_state = 1234)
scores_data = pca.fit_transform(data_vertexwise_pca)
loadings_data = (pca.components_.T * np.sqrt(pca.explained_variance_))

# Save components
for i in range(num_components):
    type_data = 'cortex' if name in ['myelin', 'thickness', 'sulc', 'curvature'] else 'cortex_subcortex'
    save_as_dscalar_and_npy(scores_data[:, i], type_data, path_results, f'{name}_PCscore_{i}')

np.save(path_results + name + '_PCscore.npy', scores_data)
np.save(path_results + name + '_PCloading.npy', loadings_data)

# Plot each loading component vs. age
for i in range(num_components):
    plt.figure(figsize = (10, 5))
    cmap = plt.get_cmap('coolwarm')
    colors = cmap(np.linspace(0, 1, 100))
    colors = [colors[90]  if s == 'F' else colors[10] for s in sex]
    plt.scatter(age,
                loadings_data[:, i],
                color = colors,
                s = 15,
                alpha = 0.5)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    if i == 0:
        plt.ylim(-0.7,0.1)
        plt.yticks(np.arange(0.1, -0.71, -0.1))
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams.update({'font.size': 8})
    plt.tight_layout()
    plt.savefig(path_figures + 'pca_' + name + '_' + str(i) + '.svg',
                format = 'svg')
    plt.show()
    print(i)
    print(spearmanr(age, loadings_data[:, i]))
    print('--------------')

# Plot the first pc score - while inverting
plt.figure(figsize = (10, 5))
cmap = plt.get_cmap('coolwarm')
colors = cmap(np.linspace(0, 1, 100))
colors = [colors[90]  if s == 'F' else colors[10] for s in sex]
plt.scatter(age,
            -1*loadings_data[:, 0],
            color = colors,
            s = 15,
            alpha = 0.5)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.ylim(-0.1, 0.7)
plt.yticks(np.arange(-0.1, 0.71, 0.1))
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({'font.size': 8})
plt.tight_layout()
plt.savefig(path_figures + 'pca_' + name + '_0_inverted.svg',
            format = 'svg')
plt.show()

# Plot variance explained
plt.figure(figsize = (5, 5))
plt.scatter(range(num_components),
            pca.explained_variance_ratio_,
            color = 'gray',
            s = 15)
plt.xlim(-0.1, 5)
plt.ylim(0, 0.6)
plt.xticks(np.arange(0, 5, 1))
plt.yticks(np.arange(0, 0.65, 0.1))
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({'font.size': 8})
plt.tight_layout()
plt.savefig(path_figures + 'pca_' + name + '_variance_explained.svg',
            format = 'svg')
plt.show()

#------------------------------------------------------------------------------
# Interaction of loading with age in each dataset
#------------------------------------------------------------------------------

dev_indices = np.arange(len(agedev))  # assuming dev subjects are listed first
aging_indices = np.arange(len(agedev), len(age))  # rest are aging subjects

# Extract loadings for each group (developmental and aging)
loadings_dev = loadings_data[dev_indices, :]
loadings_aging = loadings_data[aging_indices, :]

# Calculate Spearman correlations between age and each loading component forthe developmental group
print("Development Group:")
for i in range(num_components):
    correlation, pvalue = spearmanr(agedev, loadings_dev[:, i])
    print(f"PC {i}: Spearman correlation = {correlation}, p-value = {pvalue}")
    print("--------------")

# Calculate Spearman correlations between age and each loading component for the aging group
print("Aging Group:")
for i in range(num_components):
    correlation, pvalue = spearmanr(ageold, loadings_aging[:, i])
    print(f"PC {i}: Spearman correlation = {correlation}, p-value = {pvalue}")
    print("--------------")

#------------------------------------------------------------------------------
# END
