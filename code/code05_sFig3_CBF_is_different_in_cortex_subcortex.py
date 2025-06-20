"""

# Statistical comparison of cerebral blood perfusion between cortex and subcortex

This script performs a statistical comparison of perfusion values between cortical and subcortical regions in male
and female subjects. It applies both parametric (T-test) and non-parametric (Mann-Whitney U test) statistical methods
to determine whether there are significant differences in perfusion between the cortex and subcortex for men and women
separately. Additionally, it visualizes the perfusion distribution for both areas.


## Results:
- **Number of Subjects:**
  - Total: 1305
  - Males: 587
  - Females: 718

    T-test results-female: t-statistic = 30.653488295078947, p-value = 3.9639556227076807e-159
    T-test results-men: t-statistic = 22.95947724570251, p-value = 1.25420305036908e-96

    Mann-Whitney U test results-women: U-statistic = 452237.0, p-value = 2.991614370090351e-135
    Mann-Whitney U test results-men: U-statistic = 286287.0, p-value = 9.1565504736566e-86

Note: Related to Fig.S3.

"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import globals
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
# Load subject information and illustrate some demographics
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
print(len(sex))
print('number of male subjects')
print(sum(sex == 'M'))
print('number of female subjects')
print(sum(sex == 'F'))

#------------------------------------------------------------------------------
# Load data (vertex/voxel-wise) and stratify based on cortex/sub-cortex
#------------------------------------------------------------------------------

name = 'perfusion'

old_data_vertexwise_cortex = np.load(path_results + name + '_all_vertex.npy')[:globals.num_cort_vertices_noMW,:]
dev_data_vertexwise_cortex = np.load(path_results + 'Dev_' + name + '_all_vertex.npy')[:globals.num_cort_vertices_noMW,:]
data_vertexwise_cortex = np.concatenate((dev_data_vertexwise_cortex, old_data_vertexwise_cortex), axis = 1)

old_data_vertexwise_subcortex = np.load(path_results + name + '_all_vertex.npy')[globals.num_cort_vertices_noMW:,:]
dev_data_vertexwise_subcortex = np.load(path_results + 'Dev_' + name + '_all_vertex.npy')[globals.num_cort_vertices_noMW:,:]
data_vertexwise_subcortex  = np.concatenate((dev_data_vertexwise_subcortex, old_data_vertexwise_subcortex), axis = 1)

#------------------------------------------------------------------------------
# Stratify the cortical/sub-cortical data based on sex
#------------------------------------------------------------------------------

data_cortex_men = data_vertexwise_cortex[:, (sex == 'M')]
data_subcortex_men = data_vertexwise_subcortex[:, (sex == 'M')]

data_cortex_women = data_vertexwise_cortex[:, (sex == 'F')]
data_subcortex_women = data_vertexwise_subcortex[:, (sex == 'F')]

# Compute mean perfusion per participant for each group
data_cortex_men_singlePoint = np.mean(data_cortex_men, axis = 0)
data_cortex_women_singlePoint  = np.mean(data_cortex_women, axis = 0)

data_subcortex_men_singlePoint = np.mean(data_subcortex_men, axis = 0)
data_subcortex_women_singlePoint  = np.mean(data_subcortex_women, axis = 0)

#------------------------------------------------------------------------------
# Statistical test
#------------------------------------------------------------------------------

# Perform a t-test to compare perfusion between cortex/sub-cortex for women
t_stat_women, p_val_women = ttest_ind(data_cortex_women_singlePoint,
                                        data_subcortex_women_singlePoint,
                                        alternative = 'two-sided')
print(f"T-test results-female: t-statistic = {t_stat_women}, p-value = {p_val_women}")

# Perform a t-test to compare perfusion between cortex/sub-cortex for men
t_stat_men, p_val_men = ttest_ind(data_cortex_men_singlePoint,
                                        data_subcortex_men_singlePoint,
                                        alternative = 'two-sided')
print(f"T-test results-men: t-statistic = {t_stat_men}, p-value = {p_val_men}")

# Perform Mann-Whitney U test for non-parametric comparison for women
u_stat_women, u_p_val_women = mannwhitneyu(data_cortex_women_singlePoint,
                                             data_subcortex_women_singlePoint,
                                             alternative = 'two-sided')
print(f"Mann-Whitney U test results-women: U-statistic = {u_stat_women}, p-value = {u_p_val_women}")

# Perform Mann-Whitney U test for non-parametric comparison for men
u_stat_men, u_p_val_men = mannwhitneyu(data_cortex_men_singlePoint,
                                                   data_subcortex_men_singlePoint,
                                                   alternative = 'two-sided')
print(f"Mann-Whitney U test results-men: U-statistic = {u_stat_men}, p-value = {u_p_val_men}")

#------------------------------------------------------------------------------
# Plot bar plot with individual points corresponding to individual participants
#------------------------------------------------------------------------------

# Visualization 1: Separate plots for men and women
y_limits = (0, 155) # check this

# For male; create a dataFrame for easier plotting
plot_data = pd.DataFrame({
    'Perfusion_men': np.concatenate([data_cortex_men_singlePoint,
                                     data_subcortex_men_singlePoint]),
    'area': ['cortex'] * len(data_cortex_men_singlePoint) + \
        ['subcortex'] * len(data_subcortex_men_singlePoint)})
c1 = 'gray'
c2 = 'black'
plt.figure(figsize = (5, 5))
sns.stripplot(x = 'area',
              y = 'Perfusion_men',
              data = plot_data,
              palette = [c1, c2],
              size = 5,
              alpha = 0.7,
              jitter = True)
plt.ylim(y_limits)  # Set consistent y-axis limits
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(path_figures + 'distribution_cortex_subcortex_perfusion_men.svg',
            format = 'svg')
plt.show()

# For females
plot_data = pd.DataFrame({
    'Perfusion_women': np.concatenate([data_cortex_women_singlePoint,
                                     data_subcortex_women_singlePoint]),
    'area': ['cortex'] * len(data_cortex_women_singlePoint) + \
        ['subcortex'] * len(data_subcortex_women_singlePoint)})
c1 = 'gray'
c2 = 'black'
plt.figure(figsize = (5, 5))
sns.stripplot(x = 'area',
              y = 'Perfusion_women',
              data = plot_data,
              palette = [c1, c2],
              size = 5,
              alpha = 0.7,
              jitter = True)
plt.ylim(y_limits)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(path_figures + 'distribution_cortex_subcortex_perfusion_women.svg',
            format = 'svg')
plt.show()

# Visualization 2: combined plot for both sexes
combined_data = pd.DataFrame({
    'Perfusion': np.concatenate([data_cortex_men_singlePoint, data_subcortex_men_singlePoint,
                                 data_cortex_women_singlePoint, data_subcortex_women_singlePoint]),
    'Area': ['Cortex'] * len(data_cortex_men_singlePoint) + ['Subcortex'] * len(data_subcortex_men_singlePoint) +
            ['Cortex'] * len(data_cortex_women_singlePoint) + ['Subcortex'] * len(data_subcortex_women_singlePoint),
    'Sex': ['Men'] * (len(data_cortex_men_singlePoint) + len(data_subcortex_men_singlePoint)) +
           ['Women'] * (len(data_cortex_women_singlePoint) + len(data_subcortex_women_singlePoint))})

# Use colors based on sex for individual data points.
cmap = plt.get_cmap('coolwarm')
colors = cmap(np.linspace(0, 1, 100))
colorsS = [colors[90]  if s == 'F' else colors[10] for s in sex]
plt.figure(figsize = (8, 6))
sns.barplot(x = 'Area',
            y = 'Perfusion',
            hue = 'Sex',
            data = combined_data,
            ci = "sd",
            palette = [colors[10], colors[90]])
sns.stripplot(x = 'Area',
              y = 'Perfusion',
              data = combined_data,
              hue = 'Sex',
              dodge = True,
              jitter = True,
              alpha = 0.6,
              size = 4,
              palette = {'Men': '#1f77b4', 'Women': '#d62728'},
              marker = 'o',
              linewidth = 0.5)
plt.xlabel("Brain Area")
plt.ylabel("Perfusion")
plt.title("Perfusion in Cortex and Subcortex for Men and Women")
plt.legend(title = "Sex")
plt.ylim(y_limits)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(path_figures + 'distribution_cortex_subcortex_perfusion_bothsex.svg',
            format = 'svg')
plt.show()

#------------------------------------------------------------------------------
# END
