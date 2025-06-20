"""

Number of subjects per biological sex in each age-bin:

   Age Bin  Men  Women
0      5-9   23     34
1     9-10   20     39
2    10-11   24     22
3    11-12   19     16
4    12-13   20     24
5    13-15   64     62
6    15-17   42     46
7    17-19   27     36
8    19-21   33     36
9    21-23   18     22
10   29-35    0      0
11   35-47   67     98
12   47-57   65     88
13   57-67   58     66
14   67-77   47     61
15     77+   60     68

Assess the significance of CBF measure across male and female per age bin:

Age Bin: 5-9
T-statistic: -0.8088, P-value: 4.2529e-01
  No significant difference (p >= 0.05)

Age Bin: 9-10
T-statistic: -0.5507, P-value: 5.8469e-01
  No significant difference (p >= 0.05)

Age Bin: 10-11
T-statistic: -0.9784, P-value: 3.3330e-01
  No significant difference (p >= 0.05)

Age Bin: 11-12
T-statistic: -0.4404, P-value: 6.6357e-01
  No significant difference (p >= 0.05)

Age Bin: 12-13
T-statistic: 0.5778, P-value: 5.6653e-01
  No significant difference (p >= 0.05)

Age Bin: 13-15
T-statistic: -2.3889, P-value: 1.8474e-02
  ** Significant difference between men and women (p < 0.05) **

Age Bin: 15-17
T-statistic: -4.7678, P-value: 7.5291e-06
  ** Significant difference between men and women (p < 0.05) **

Age Bin: 17-19
T-statistic: -4.3597, P-value: 5.2314e-05
  ** Significant difference between men and women (p < 0.05) **

Age Bin: 19-21
T-statistic: -4.3922, P-value: 4.1910e-05
  ** Significant difference between men and women (p < 0.05) **

Age Bin: 21-23
T-statistic: -4.0244, P-value: 2.6888e-04
  ** Significant difference between men and women (p < 0.05) **

Age Bin: 29-35
T-statistic: nan, P-value: nan
  No significant difference (p >= 0.05)

Age Bin: 35-47
T-statistic: -6.7027, P-value: 3.9130e-10
  ** Significant difference between men and women (p < 0.05) **

Age Bin: 47-57
T-statistic: -6.7457, P-value: 3.0873e-10
  ** Significant difference between men and women (p < 0.05) **

Age Bin: 57-67
T-statistic: -5.4027, P-value: 3.5573e-07
  ** Significant difference between men and women (p < 0.05) **

Age Bin: 67-77
T-statistic: -3.6007, P-value: 4.8620e-04
  ** Significant difference between men and women (p < 0.05) **

Age Bin: 77+
T-statistic: -3.9912, P-value: 1.1106e-04
  ** Significant difference between men and women (p < 0.05) **

Best n_spline: 9 with MSE: 176.8500

Note: Related to Fig.1 & Fig.4a & Table.S4.

"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
from pygam import LinearGAM, s
import matplotlib.pyplot as plt
from IPython import get_ipython
from sklearn.model_selection import KFold
from globals import path_figures, path_info_sub, path_results

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------

warnings.filterwarnings("ignore")
get_ipython().magic('reset -sf')

#------------------------------------------------------------------------------
# Load subject information
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
# Load data (vertex-wise)
#------------------------------------------------------------------------------

# Load aging and developmental subjects data
old_data_vertexwise = np.load(path_results + 'perfusion_all_vertex.npy')
dev_data_vertexwise = np.load(path_results + 'Dev_perfusion_all_vertex.npy')
data_vertexwise = np.concatenate((dev_data_vertexwise, old_data_vertexwise), axis = 1)

#------------------------------------------------------------------------------
# Stratify data by sex
#------------------------------------------------------------------------------

data_men = data_vertexwise[:, sex == 'M']
data_women = data_vertexwise[:, sex == 'F']

age_men = age[sex == 'M']
age_women = age[sex == 'F']

#------------------------------------------------------------------------------
# Plot sex by total CBF scatter plot - colored based on biological sex of participants
#------------------------------------------------------------------------------

cmap = plt.get_cmap('coolwarm')
colors = cmap(np.linspace(0, 1, 100))
plt.figure(figsize = (10, 5))
plt.scatter(age_men,
            np.mean(data_men, axis = 0),
            alpha = 0.7,
            color =  colors[10])
plt.scatter(age_women,
            np.mean(data_women, axis = 0),
            alpha = 0.7,
            color =  colors[90])
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({'font.size': 8})
plt.tight_layout()
plt.savefig(path_figures + 'mean_CBF_lifespan.svg',
            format = 'svg')
plt.show()

#------------------------------------------------------------------------------
# Group participants into bins based on age-sex and see when changes between men and women emerge
#------------------------------------------------------------------------------

age_bins = [0, 9, 10, 11, 12, 13, 15, 17, 19, 21, 23, 35, 47, 57, 67, 77, 100]
age_labels = ['5-9', '9-10', '10-11', '11-12', '12-13',
              '13-15', '15-17', '17-19', '19-21', '21-23',
              '29-35', '35-47', '47-57', '57-67', '67-77', '77+']

# Apply binning on the age data for men and women
age_bins_men = pd.cut(age_men, bins = age_bins, labels = age_labels)
age_bins_women = pd.cut(age_women, bins = age_bins, labels = age_labels)

# Combine data into a single DataFrame for easy plotting
data_combined = pd.DataFrame({
    'Age Group': np.concatenate((age_bins_men,
                                 age_bins_women)),
    'Mean Perfusion': np.concatenate((np.mean(data_men, axis = 0),
                                      np.mean(data_women, axis = 0))),
    'Sex': ['Men'] * len(age_bins_men) + ['Women'] * len(age_bins_women)})

# Apply binning on the age data for men and women separately
age_bins_men = pd.cut(age[sex == 'M'],
                      bins = age_bins,
                      labels = age_labels)
age_bins_women = pd.cut(age[sex == 'F'],
                        bins = age_bins,
                        labels = age_labels)

# Count number of subjects per bin for each sex
subjects_per_bin = pd.DataFrame({
    'Age Bin': age_labels,
    'Men': age_bins_men.value_counts().sort_index().values,
    'Women': age_bins_women.value_counts().sort_index().values})
print(subjects_per_bin)

plt.figure(figsize = (10, 5))
sns.stripplot(x = 'Age Group',
              y = 'Mean Perfusion',
              hue = 'Sex',
              data = data_combined,
              dodge = True,
              jitter = True,
              alpha = 0.7,
              size = 5,
              order = age_labels,
              palette = {'Men': colors[10], 'Women': colors[90]})

plt.title('Mean Perfusion Across Age Bins (Men and Women)')
plt.xlabel('Age Bins')
plt.ylabel('Mean Perfusion')
plt.xticks(rotation = 45)
plt.legend(title='Sex')
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.tight_layout()
plt.savefig(path_figures + 'mean_binarized_sex_difference_CBF_lifespan.svg',
            format = 'svg')
plt.show()

#------------------------------------------------------------------------------
# Create the same figure but this time with statistics included
#------------------------------------------------------------------------------

# Perform t-test for each bin and store results
significant_bins = [] # To store the bins that are significant
for bin_label in age_labels:

    # Get data for men and women for the current bin
    men_data = data_combined[(data_combined['Age Group'] == bin_label) \
                             & (data_combined['Sex'] == 'Men')]['Mean Perfusion']
    women_data = data_combined[(data_combined['Age Group'] == bin_label) \
                               & (data_combined['Sex'] == 'Women')]['Mean Perfusion']

    # Perform t-test (assuming unequal variance with `equal_var=False`)
    t_stat, p_value = stats.ttest_ind(men_data,
                                      women_data,
                                      equal_var = False)

    # Print the results for this bin
    print(f"Age Bin: {bin_label}")
    print(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4e}")
    if p_value < 0.05:
        print("  ** Significant difference between men and women (p < 0.05) **\n")
        significant_bins.append(bin_label) # Store the bin label for significant results
    else:
        print("  No significant difference (p >= 0.05)\n")

# Plot the data
plt.figure(figsize = (10, 5))
sns.stripplot(x = 'Age Group',
              y = 'Mean Perfusion',
              hue = 'Sex',
              data = data_combined,
              dodge = True,
              jitter = True,
              alpha = 0.7,
              size = 5,
              order = age_labels,
              palette = {'Men': colors[10], 'Women': colors[90]})

# Add asterisks (*) for significant bins
for bin_label in significant_bins:
    men_data = data_combined[(data_combined['Age Group'] == bin_label) & (data_combined['Sex'] == 'Men')]['Mean Perfusion']
    women_data = data_combined[(data_combined['Age Group'] == bin_label) & (data_combined['Sex'] == 'Women')]['Mean Perfusion']
    y_max = max(men_data.max(), women_data.max())
    x_position = age_labels.index(bin_label)
    plt.text(x_position,
             y_max + 0.05,
             '*',
             ha = 'center',
             va = 'bottom',
             color = 'black',
             fontsize = 16)

# Final layout adjustment and show the plot
plt.title('Mean Perfusion Across Age Bins (Men and Women)')
plt.xlabel('Age Bins')
plt.ylabel('Mean Perfusion')
plt.xticks(rotation = 45)
plt.legend(title = 'Sex')
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.tight_layout()
plt.savefig(path_figures + 'mean_binarized_sex_difference_CBF_lifespan_withstatistics.svg',
            format = 'svg')
plt.show()

#------------------------------------------------------------------------------
# Fit a line into the trajectories of whole-brain CBF change
#------------------------------------------------------------------------------

# Find the best number of splines
n_splines_range = range(4, 15, 1)  # Adjust the range and step as needed
best_n_spline = None
best_score = float('inf')

# Cross-validation setup
kf = KFold(n_splits = 5, shuffle = True, random_state = 42)

# Loop over each `n_spline` value
for n_splines in n_splines_range:
    scores = []

    # Cross-validate for each `n_spline`
    for train_idx, test_idx in kf.split(age_men):
        # Split the data
        age_train, age_test = age_men[train_idx], age_men[test_idx]
        data_train, data_test = np.mean(data_men[:, train_idx], axis = 0), np.mean(data_men[:, test_idx], axis = 0)

        # Fit the GAM model
        gam = LinearGAM(s(0, n_splines = n_splines)).fit(age_train, data_train)

        # Calculate the test MSE
        test_pred = gam.predict(age_test)
        mse = np.mean((data_test - test_pred) ** 2)
        scores.append(mse)

    # Calculate the mean cross-validated score
    mean_score = np.mean(scores)

    # Update the best score and `n_spline` if this one is better
    if mean_score < best_score:
        best_score = mean_score
        best_n_spline = n_splines

# Print the best `n_spline`
print(f'Best n_spline: {best_n_spline} with MSE: {best_score:.4f}')

# Fit GAM for data
gam_men = LinearGAM(s(0, n_splines = best_n_spline)).fit(age_men, np.mean(data_men, axis = 0))
gam_women = LinearGAM(s(0, n_splines = best_n_spline)).fit(age_women, np.mean(data_women, axis = 0))

# Predict on age range for smooth curves
age_range = np.linspace(age.min(), age.max(), 500)
pred_men = gam_men.predict(age_range)
pred_women = gam_women.predict(age_range)

# Calculate the difference between the GAM predictions
difference = pred_men - pred_women

# Plot the GAM fits and the difference in subplots
fig, axs = plt.subplots(2, 1, figsize = (12, 10), sharex = True)

# Plot GAM fits for men and women
axs[0].plot(age_range,
            pred_men,
            label = 'GAM Fit - Men',
            color = colors[0],
            linewidth = 5)
axs[0].plot(age_range,
            pred_women,
            label = 'GAM Fit - Women',
            color = colors[99],
            linewidth = 5)
axs[0].scatter(age_men, np.mean(data_men, axis = 0),
               alpha = 0.6,
               color = colors[10],
               label = 'Data - Men')
axs[0].scatter(age_women,
               np.mean(data_women, axis = 0),
               alpha = 0.6,
               color = colors[90],
               label = 'Data - Women')
axs[0].set_title('GAM Fit for Mean Perfusion Across Age (Men and Women)')
axs[0].set_ylabel('Mean Perfusion')
axs[0].legend()
axs[0].spines['right'].set_visible(False)
axs[0].spines['top'].set_visible(False)

# Plot the difference curve
axs[1].plot(age_range, difference,
            color = 'black',
            linewidth = 5)

# Identify the min and max points on the difference curve
min_diff_idx = np.argmin(difference)
max_diff_idx = np.argmax(difference)
min_diff_age = age_range[min_diff_idx]
max_diff_age = age_range[max_diff_idx]
min_diff_value = difference[min_diff_idx]
max_diff_value = difference[max_diff_idx]

# Annotate min and max values on the difference curve
axs[1].plot(min_diff_age, min_diff_value, 'bo', linewidth = 5)
axs[1].plot(max_diff_age, max_diff_value, 'ro', linewidth = 5)
axs[1].vlines([min_diff_age,
               max_diff_age],
              min_diff_value,
              max_diff_value,
              colors = 'gray',
              linestyles = 'dotted',
              linewidth = 5)

# Display values at min and max points
axs[1].text(min_diff_age,
            min_diff_value,
            f'{min_diff_value:.2f}',
            ha = 'right',
            color = 'blue',
            fontsize = 10)
axs[1].text(max_diff_age,
            max_diff_value,
            f'{max_diff_value:.2f}',
            ha = 'left',
            color = 'red',
            fontsize = 10)

axs[1].set_title('Difference Between Men and Women GAM Fit')
axs[1].set_xlabel('Age')
axs[1].set_ylabel('Difference in Mean Perfusion')
axs[1].spines['right'].set_visible(False)
axs[1].spines['top'].set_visible(False)

plt.tight_layout()
plt.savefig(path_figures + 'fit_line_men_women_CBF_total_brain.svg',
            format = 'svg')
plt.show()
#------------------------------------------------------------------------------
# END