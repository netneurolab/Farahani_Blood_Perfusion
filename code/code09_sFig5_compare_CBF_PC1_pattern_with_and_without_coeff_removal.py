"""

Extract the pattern of perfusion across participants using residualized data.
(i.e., regressed out for age and sex and thier non-linear affects).

Comparison with the main method shows high similarity, e.g.

  rho = -0.9310911299303585
  r = -0.9304260629899258

The first principal component explains 72.8% of the variance in the data.

Note: Related to Fig.S5.

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
from scipy.stats import zscore, spearmanr, pearsonr
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

# Here we load two dataframes:
#   1) Adult/aging data (clean_data_info.csv)
#   2) Development data (Dev_clean_data_info.csv)

df = pd.read_csv(path_info_sub + 'clean_data_info.csv')
num_subjects = len(df)
ageold = np.array(df['interview_age'])/12
sexold = df.sex

df = pd.read_csv(path_info_sub + 'Dev_clean_data_info.csv')
num_subjects = len(df)
agedev = np.array(df['interview_age'])/12
sexdev = df.sex

# Concatenate age and sex from both datasets
sex = np.concatenate((sexdev, sexold))
age = np.concatenate((agedev, ageold))

#------------------------------------------------------------------------------
# Load data (vertex-wise) - cleaned datasets
#------------------------------------------------------------------------------

data_vertexwise = np.load(path_results + 'perfusion_clean_sex_age_2datasets_GLM.npy')

#------------------------------------------------------------------------------
# Perform PCA - vertex wise - cortex and subcortex
#------------------------------------------------------------------------------

data_vertexwise_pca = zscore(data_vertexwise, axis = 1)
data_vertexwise_pca = zscore(data_vertexwise, axis = 0)
num_components = 5
pca = PCA(n_components = num_components)
scores_data = pca.fit_transform(data_vertexwise_pca)
loadings_data = (pca.components_.T * np.sqrt(pca.explained_variance_))

# Save components
for i in range(num_components):
    type_data = 'cortex_subcortex'
    save_as_dscalar_and_npy(scores_data[:, i], type_data, path_results, f'perfusion_regressedout_PCscore_{i}')

np.save(path_results + 'perfusion_regressedout_PCscore.npy', scores_data)
np.save(path_results + 'perfusion_regressedout_PCloading.npy', loadings_data)

# Scatter plots of each PCA component's loadings vs. age
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
        plt.ylim(-1, -0.2)
        #plt.yticks(np.arange(-0.2, -1.01, 0.1))
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams.update({'font.size': 8})
    plt.tight_layout()
    plt.savefig(path_figures + 'pca_perfusion_regressedout_' + str(i) + '.svg',
                format = 'svg')
    plt.show()
    print(i)
    print(spearmanr(age, loadings_data[:, i]))
    print('--------------')

# Scatter plots of first PCA loadings - inverted (principal component  directionality is reversible)
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
plt.ylim(0.3, 1)
plt.yticks(np.arange(0.3, 1.01, 0.1))
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({'font.size': 8})
plt.tight_layout()
plt.savefig(path_figures + 'pca_perfusion_regressedout_0_inverted.svg',
            format = 'svg')
plt.show()

# Plot the variance explained by each principal component
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
plt.savefig(path_figures + 'pca_perfusion_regressedout_variance_explained.svg',
            format = 'svg')
plt.show()

#------------------------------------------------------------------------------
# Interaction of loadings with age in each dataset (development and aging)
#------------------------------------------------------------------------------

dev_indices = np.arange(len(agedev)) # assuming dev subjects are listed first
aging_indices = np.arange(len(agedev), len(age)) # rest are aging subjects

# Extract loadings for each group
loadings_dev = loadings_data[dev_indices, :]
loadings_aging = loadings_data[aging_indices, :]

# Calculate Spearman correlations between age and each loading component for development
print("Development Group:")
for i in range(num_components):
    correlation, pvalue = spearmanr(agedev, loadings_dev[:, i])
    print(f"PC {i}: Spearman correlation = {correlation}, p-value = {pvalue}")
    print("--------------")

# Calculate Spearman correlations between age and each loading component for aging
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

# Plot bar plot with individual data points to visualize sex differences
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
plt.savefig(path_figures + 'distridution_total_Loading_0_men_women_perfusion_regressedout.svg',
            format = 'svg')
plt.show()

#------------------------------------------------------------------------------
# Compare methods: does regressing out the age and sex effect cause huge differences?
#------------------------------------------------------------------------------
# PC_main   => PC from the original data
# PC_clean  => PC from the regressed-out (age/sex) data

PC_main = np.load(path_results + 'perfusion_PCscore.npy')[:, 0]
PC_clean = np.load(path_results + 'perfusion_regressedout_PCscore.npy')[:, 0]

print('similarity of the main PC with the PC obtaiend using the regressedout data')
print(spearmanr(PC_main, PC_clean))
print(pearsonr(PC_main, PC_clean))

# Scatter plot comparison
plt.figure(figsize = (5, 5))
plt.scatter(PC_main,
            PC_clean,
            color = 'gray',
            alpha = 0.7,
            s = 0.1)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(path_figures + 'PC1_comparison_methods.png',
            format = 'png')
plt.show()

# Scatter plot comparison - inverted version
plt.figure(figsize = (5, 5))
plt.scatter(PC_main,
            -1*PC_clean,
            color = 'gray',
            alpha = 0.7,
            s = 0.1)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(path_figures + 'PC1_comparison_methods_inverted.png',
            format = 'png')
plt.show()

#------------------------------------------------------------------------------
# END