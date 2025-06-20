"""

# Analysis WM matter blood perfusion for HCP-Lifespan subjects

More details:

	Age Bin: 5-9
	T-statistic: 1.1370, P-value: 2.6252e-01
	  No significant difference (p >= 0.05)

	Age Bin: 9-10
	T-statistic: 0.1565, P-value: 8.7656e-01
	  No significant difference (p >= 0.05)

	Age Bin: 10-11
	T-statistic: -0.9534, P-value: 3.4573e-01
	  No significant difference (p >= 0.05)

	Age Bin: 11-12
	T-statistic: -0.5475, P-value: 5.8845e-01
	  No significant difference (p >= 0.05)

	Age Bin: 12-13
	T-statistic: 1.2499, P-value: 2.2007e-01
	  No significant difference (p >= 0.05)

	Age Bin: 13-15
	T-statistic: -3.7588, P-value: 2.6197e-04
	  ** Significant difference between men and women (p < 0.05) **

	Age Bin: 15-17
	T-statistic: -4.9733, P-value: 3.4327e-06
	  ** Significant difference between men and women (p < 0.05) **

	Age Bin: 17-19
	T-statistic: -1.9526, P-value: 5.8516e-02
	  No significant difference (p >= 0.05)

	Age Bin: 19-21
	T-statistic: -4.7347, P-value: 1.3613e-05
	  ** Significant difference between men and women (p < 0.05) **

	Age Bin: 21-23
	T-statistic: -2.9603, P-value: 5.2966e-03
	  ** Significant difference between men and women (p < 0.05) **

	Age Bin: 29-35
	T-statistic: nan, P-value: nan
	  No significant difference (p >= 0.05)

	Age Bin: 35-47
	T-statistic: -7.2583, P-value: 1.7536e-11
	  ** Significant difference between men and women (p < 0.05) **

	Age Bin: 47-57
	T-statistic: -6.9645, P-value: 9.4787e-11
	  ** Significant difference between men and women (p < 0.05) **

	Age Bin: 57-67
	T-statistic: -3.8562, P-value: 1.8818e-04
	  ** Significant difference between men and women (p < 0.05) **

	Age Bin: 67-77
	T-statistic: -3.3946, P-value: 9.6831e-04
	  ** Significant difference between men and women (p < 0.05) **

	Age Bin: 77+
	T-statistic: -4.6059, P-value: 1.0061e-05
	  ** Significant difference between men and women (p < 0.05) **

Note: Related to Fig.S1.

"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import nibabel as nib
import seaborn as sns
import scipy.stats as stats
from pygam import LinearGAM, s
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from globals import path_info_sub, path_wm_aging, path_wm_dev, path_figures

#------------------------------------------------------------------------------
# AGING dataset **** Get subject list - load WM data and average over parcels
#------------------------------------------------------------------------------

df_aging = pd.read_csv(path_info_sub + 'clean_data_info.csv')
subject_ids_aging = df_aging.src_subject_id

WM_average_aging = np.zeros(len(subject_ids_aging))
for n, subid in enumerate(subject_ids_aging):
    # Define paths for ASL wm data
    path_data_ASL = os.path.join(path_wm_aging,  subid + '_V1_MR/MNINonLinear/ASL/')
    file_perfusion_wm = os.path.join(path_data_ASL + 'pvcorr_perfusion_wm_calib_masked.nii.gz')
    img_perfusion_wm = nib.load(file_perfusion_wm).get_data()
    img_perfusion_wm[img_perfusion_wm == 0] = np.nan
    WM_average_aging[n] = np.nanmean(img_perfusion_wm.flatten())
    print(n)

sex_aging = df_aging['sex'] 
age_aging = df_aging['interview_age'] /12
data_men_aging = WM_average_aging[sex_aging == 'M']
data_women_aging = WM_average_aging[sex_aging == 'F']

age_men_aging = age_aging[sex_aging == 'M']
age_women_aging = age_aging[sex_aging == 'F']
age_men_aging = np.array(age_men_aging)
age_women_aging = np.array(age_women_aging)

#------------------------------------------------------------------------------
# DEVELOPMENTAL dataset **** Get subject list - load WM data and average over parcels
#------------------------------------------------------------------------------

df_dev = pd.read_csv(path_info_sub + 'Dev_clean_data_info.csv')
subject_ids_dev = df_dev.src_subject_id

WM_average_dev = np.zeros(len(subject_ids_dev))
for n, subid in enumerate(subject_ids_dev):
    # Define paths for ASL wm data
    path_data_ASL = os.path.join(path_wm_dev,  subid + '_V1_MR/MNINonLinear/ASL/')
    file_perfusion_wm = os.path.join(path_data_ASL + 'pvcorr_perfusion_wm_calib_masked.nii.gz')
    img_perfusion_wm = nib.load(file_perfusion_wm).get_data()
    img_perfusion_wm[img_perfusion_wm == 0] = np.nan
    WM_average_dev[n] = np.nanmean(img_perfusion_wm.flatten())
    print(n)

sex_dev = df_dev['sex'] 
age_dev = df_dev['interview_age'] /12
data_men_dev = WM_average_dev[sex_dev == 'M']
data_women_dev = WM_average_dev[sex_dev == 'F']

age_men_dev = age_dev[sex_dev == 'M']
age_women_dev = age_dev[sex_dev == 'F']
age_men_dev = np.array(age_men_dev)
age_women_dev = np.array(age_women_dev)

#------------------------------------------------------------------------------
# Combine all data to one another
#------------------------------------------------------------------------------

age_men = np.concatenate((age_men_dev, age_men_aging), axis = 0)
age_women = np.concatenate((age_women_dev, age_women_aging), axis = 0)

data_men = np.concatenate((data_men_dev, data_men_aging), axis = 0)
data_women = np.concatenate((data_women_dev, data_women_aging), axis = 0)

age = np.concatenate((age_dev, age_aging), axis = 0)
sex = np.concatenate((sex_dev, sex_aging), axis = 0)

#------------------------------------------------------------------------------
# From now on all is about plotting the results
#------------------------------------------------------------------------------

cmap = plt.get_cmap('coolwarm')
colors = cmap(np.linspace(0, 1, 100))
plt.figure(figsize = (10, 5))
plt.scatter(age_men,
            data_men,
            alpha = 0.7,
            color =  colors[10])
plt.scatter(age_women,
            data_women,
            alpha = 0.7,
            color =  colors[90])
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({'font.size': 8})
plt.tight_layout()
plt.show()

#------------------------------------------------------------------------------
# Fit a line into the trajectories of whole-brain CBF change
#------------------------------------------------------------------------------

# Find the best number of splines
n_splines_range = range(4, 15, 1)
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
        data_train, data_test = data_men[ train_idx],data_men[ test_idx]

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
gam_men = LinearGAM(s(0, n_splines = best_n_spline)).fit(age_men, data_men)
gam_women = LinearGAM(s(0, n_splines = best_n_spline)).fit(age_women,data_women)

# Predict on age range for smooth curves
age_range = np.linspace(age.min(), age.max(), 500)
pred_men = gam_men.predict(age_range)
pred_women = gam_women.predict(age_range)

# Calculate the difference between the GAM predictions
difference = pred_men - pred_women

# Plot the GAM fits
fig, axs = plt.subplots(2, 1, figsize = (12, 10), sharex = True)
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
axs[0].scatter(age_men, data_men,
               alpha = 0.6,
               color = colors[10],
               label = 'Data - Men')
axs[0].scatter(age_women,
               data_women,
               alpha = 0.6,
               color = colors[90],
               label = 'Data - Women')
axs[0].set_title('GAM Fit for Mean Perfusion Across Age (Men and Women)')
axs[0].set_ylabel('Mean Perfusion')
axs[0].legend()
axs[0].spines['right'].set_visible(False)
axs[0].spines['top'].set_visible(False)
axs[1].plot(age_range, difference,
            color = 'black',
            linewidth = 5)
min_diff_idx = np.argmin(difference)
max_diff_idx = np.argmax(difference)
min_diff_age = age_range[min_diff_idx]
max_diff_age = age_range[max_diff_idx]
min_diff_value = difference[min_diff_idx]
max_diff_value = difference[max_diff_idx]
axs[1].plot(min_diff_age, min_diff_value, 'bo', linewidth = 5)
axs[1].plot(max_diff_age, max_diff_value, 'ro', linewidth = 5)
axs[1].vlines([min_diff_age,
               max_diff_age],
              min_diff_value,
              max_diff_value,
              colors = 'gray',
              linestyles = 'dotted',
              linewidth = 5)
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
plt.savefig(path_figures + 'trajectory_WhiteMatter_CBF_lifespan.svg',
            format = 'svg')
plt.show()

#------------------------------------------------------------------------------
# Group participants into bins based on age-sex and see when changes between men and women emerge
#------------------------------------------------------------------------------

age_bins = [0, 9, 10, 11, 12, 13, 15, 17, 19, 21, 23, 35, 47, 57, 67, 77, 100]
age_labels = ['5-9', '9-10', '10-11', '11-12', '12-13',
              '13-15', '15-17', '17-19', '19-21', '21-23',
              '29-35', '35-47', '47-57', '57-67', '67-77', '77+']

age_bins_men = pd.cut(age_men, bins = age_bins, labels = age_labels)
age_bins_women = pd.cut(age_women, bins = age_bins, labels = age_labels)

data_combined = pd.DataFrame({
    'Age Group': np.concatenate((age_bins_men, age_bins_women), axis = 0),
    'Mean Perfusion': np.concatenate((data_men, data_women), axis = 0),
    'Sex': ['Men'] * len(age_bins_men) + ['Women'] * len(age_bins_women)})

age_bins_men = pd.cut(age[sex == 'M'],
                      bins = age_bins,
                      labels = age_labels)
age_bins_women = pd.cut(age[sex == 'F'],
                        bins = age_bins,
                        labels = age_labels)
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

plt.title('Mean Perfusion Across Age Bins (Men and Women) - white matter')
plt.xlabel('Age Bins')
plt.ylabel('Mean Perfusion')
plt.xticks(rotation = 45)
plt.legend(title='Sex')
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.tight_layout()
plt.show()

#------------------------------------------------------------------------------
# Create the same figure but this time with statistics included - white matter data
#------------------------------------------------------------------------------

# Perform t-test for each bin
significant_bins = []
for bin_label in age_labels:

    men_data = data_combined[(data_combined['Age Group'] == bin_label) \
                             & (data_combined['Sex'] == 'Men')]['Mean Perfusion']
    women_data = data_combined[(data_combined['Age Group'] == bin_label) \
                               & (data_combined['Sex'] == 'Women')]['Mean Perfusion']

    t_stat, p_value = stats.ttest_ind(men_data,
                                      women_data,
                                      equal_var = False)
    print(f"Age Bin: {bin_label}")
    print(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4e}")
    if p_value < 0.05:
        print("  ** Significant difference between men and women (p < 0.05) **\n")
        significant_bins.append(bin_label)
    else:
        print("  No significant difference (p >= 0.05)\n")

# Save to CSV
summary_stats.to_csv(path_figures + 'summary_stats_CBF_WhiteMatter_lifespan.csv', index=False)

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
plt.title('Mean Perfusion Across Age Bins (Men and Women)- with matter')
plt.xlabel('Age Bins')
plt.ylabel('Mean Perfusion')
plt.xticks(rotation = 45)
plt.legend(title = 'Sex')
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.tight_layout()
plt.savefig(path_figures + 'mean_binarized_sex_difference_CBF_WhiteMatter_lifespan_withstatistics.svg',
            format = 'svg')
plt.show()
#------------------------------------------------------------------------------
# END
