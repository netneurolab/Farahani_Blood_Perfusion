"""

# In general, the female brain has higher cerebral blood perfusion than the male brain.

Demographic information of included subjects:
    number of subjects
    1305
    number of male subjects
    587
    number of female subjects
    718

Statistical test results:
    T-test results: t-statistic = 9.274049734288107, p-value = 7.206236797826273e-20
    Mann-Whitney U test results: U-statistic = 274350.0, p-value = 5.827668838503535e-21

Note: Related to Fig.S2b (left side).

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
from scipy.stats import ttest_ind, mannwhitneyu
from globals import path_info_sub, path_results, path_figures

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------

warnings.filterwarnings("ignore")
get_ipython().magic('reset -sf')

#------------------------------------------------------------------------------
# Load perfusion data (development + aging)
#------------------------------------------------------------------------------

name = 'perfusion'

# Load aging perfusion data
old_data_vertexwise = np.load(path_results + name + '_all_vertex.npy')

# Load developmental perfusion data
dev_data_vertexwise = np.load(path_results + 'Dev_' + name + '_all_vertex.npy')

# Concatenate data from both cohorts
data_vertexwise = np.concatenate((dev_data_vertexwise, old_data_vertexwise), axis = 1)

#------------------------------------------------------------------------------
# Load subject information and show some demographics
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

# Print demographic information
print('number of subjects')
print(len(age))
print('number of male subjects')
print(sum(sex == 'M'))
print('number of female subjects')
print(sum(sex == 'F'))

#------------------------------------------------------------------------------
# Get vertex/voxel-wise data and stratify the data based on sex
#------------------------------------------------------------------------------

data_men = data_vertexwise[:, (sex == 'M')]
data_women = data_vertexwise[:, (sex == 'F')]

data_men_singlePoint  = np.mean(data_men, axis = 0)
data_women_singlePoint  = np.mean(data_women, axis = 0)

#------------------------------------------------------------------------------
# Statistical test - are female and male different in terms of blood perfusion?
#------------------------------------------------------------------------------

# Perform a t-test to compare perfusion between men and women
t_stat, p_val = ttest_ind(data_women_singlePoint, data_men_singlePoint, alternative = 'two-sided')
print(f"T-test results: t-statistic = {t_stat}, p-value = {p_val}")

# Mann-Whitney U test for non-parametric comparison
u_stat, u_p_val = mannwhitneyu(data_women_singlePoint, data_men_singlePoint, alternative = 'two-sided')
print(f"Mann-Whitney U test results: U-statistic = {u_stat}, p-value = {u_p_val}")

#------------------------------------------------------------------------------
# Plot individual data points (dots), each dot is the mean blood perfusion across all brain
#------------------------------------------------------------------------------

plot_data = pd.DataFrame({
    'Perfusion': np.concatenate([data_men_singlePoint, data_women_singlePoint]),
    'Sex': ['Men'] * len(data_men_singlePoint) + ['Women'] * len(data_women_singlePoint)})

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
plt.savefig(path_figures + 'distribution_total_perfusion_men_women.svg',
            format = 'svg')
plt.show()
#------------------------------------------------------------------------------
# END
