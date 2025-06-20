"""

# Get the main pattern of cerebral blood perfusion across participants.

We first z-score data of individual subjects and then apply PCA on the data.

Results of the PCA:

------------------------------------------

pca.explained_variance_ratio_
      0.50705391,
      0.02351177,
      0.01716106,
      0.01075986,
      0.00702661

------------------------------------------

Get the correlation of loadings with age: Lifespan
0
SpearmanrResult(correlation=-0.4635232554632101, pvalue=1.7273269761730185e-70)
--------------
1
SpearmanrResult(correlation=0.8245539379070986, pvalue=0.0)
--------------
2
SpearmanrResult(correlation=-0.25162099991461445, pvalue=2.7078198203612445e-20)
--------------
3
SpearmanrResult(correlation=0.21446654432903228, pvalue=4.823469158844423e-15)
--------------
4
SpearmanrResult(correlation=0.057730182217585004, pvalue=0.03704833045536811)

------------------------------------------

Get the correlation of loadings with age: Development group

PC 0: Spearman correlation = -0.21648456004692587, p-value = 4.374989965952371e-08
--------------
PC 1: Spearman correlation = 0.524535459990045, p-value = 1.2946134957768275e-45
--------------
PC 2: Spearman correlation = -0.2481548652138748, p-value = 2.972645111703059e-10
--------------
PC 3: Spearman correlation = -0.023641441369324456, p-value = 0.5545993162405793
--------------
PC 4: Spearman correlation = -0.008801687275922333, p-value = 0.825903519477338

------------------------------------------

Get the correlation of loadings with age: Aging group

PC 0: Spearman correlation = -0.3310796036611663, p-value = 8.304992996962646e-19
--------------
PC 1: Spearman correlation = 0.54939432451994, p-value = 1.0080118641047661e-54
--------------
PC 2: Spearman correlation = 0.1981813159656047, p-value = 1.9662932063729147e-07
--------------
PC 3: Spearman correlation = -0.05354779246058738, p-value = 0.16370231804026572
--------------
PC 4: Spearman correlation = 0.10607376946829464, p-value = 0.005697486180142085

------------------------------------------

# Loading Sex differences:
    T-test results: t-statistic = 7.660956937383907, p-value = 3.5834087983129206e-14
    Mann-Whitney U test results: U-statistic = 261663.0, p-value = 5.48536739465301e-14

Note: Related to Fig.2a,b & Fig.S4a.

"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython import get_ipython
from sklearn.decomposition import PCA
from scipy.stats import zscore, spearmanr
from functions import save_as_dscalar_and_npy
from scipy.stats import ttest_ind, mannwhitneyu
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

name = 'perfusion'

old_data_vertexwise = np.load(path_results + name + '_all_vertex.npy')
dev_data_vertexwise = np.load(path_results + 'Dev_' + name + '_all_vertex.npy')
data_vertexwise = np.concatenate((dev_data_vertexwise, old_data_vertexwise), axis = 1)

#------------------------------------------------------------------------------
# Perform PCA - vertex wise - cortex and subcortex
#------------------------------------------------------------------------------

data_vertexwise_pca = zscore(data_vertexwise, axis = 0)
num_components = 5
pca = PCA(n_components = num_components)
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
        plt.ylim(-0.3, 0.9)
        plt.yticks(np.arange(-0.3, 0.91, 0.1))
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
# Is there sex-biases in the loadings of the first PC?
#------------------------------------------------------------------------------

female_loadings_data = loadings_data[sex == 'F', 0]
male_loadings_data = loadings_data[sex == 'M', 0]

# Perform a t-test to compare perfusion between men and women
t_stat, p_val = ttest_ind(female_loadings_data, male_loadings_data, alternative = 'two-sided')
print(f"T-test results: t-statistic = {t_stat}, p-value = {p_val}")

# Mann-Whitney U test for non-parametric comparison
u_stat, u_p_val = mannwhitneyu(female_loadings_data, male_loadings_data, alternative = 'two-sided')
print(f"Mann-Whitney U test results: U-statistic = {u_stat}, p-value = {u_p_val}")

# Plot bar plot with individual data points (dots)
plot_data = pd.DataFrame({
    'Perfusion': np.concatenate([male_loadings_data, female_loadings_data]),
    'Sex': ['Men'] * len(male_loadings_data) + ['Women'] * len(female_loadings_data)
})

cmap = plt.get_cmap('coolwarm')
colors = cmap(np.linspace(0, 1, 100))
plt.figure(figsize = (5, 5))
sns.stripplot(x = 'Sex',
              y = 'Perfusion',
              data = plot_data,
              palette = [colors[10], colors[90]],
              size = 5,
              alpha = 0.7,
              jitter = True)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(path_figures + 'distribution_Perfusion_Loading_PC0_by_sex.svg',
            format = 'svg')
plt.show()

#------------------------------------------------------------------------------
# END
