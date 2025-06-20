"""

GAM applied to blood perfusion trajectory in HCP-Development

===== Within-Male Yeo Class Comparisons =====
ANOVA: F = 26.76, p = 9.9042e-27

Pairwise Yeo comparisons (t-tests, FDR-corrected):
Yeo 1 vs Yeo 2: raw p = 3.2816e-02, corrected p = 4.3071e-02 *
Yeo 1 vs Yeo 3: raw p = 5.4898e-05, corrected p = 1.6469e-04 *
Yeo 1 vs Yeo 4: raw p = 2.0571e-04, corrected p = 4.3199e-04 *
Yeo 1 vs Yeo 5: raw p = 5.7723e-01, corrected p = 6.0610e-01 
Yeo 1 vs Yeo 6: raw p = 5.9950e-08, corrected p = 2.0983e-07 *
Yeo 1 vs Yeo 7: raw p = 4.2699e-10, corrected p = 2.7640e-09 *
Yeo 2 vs Yeo 3: raw p = 5.2647e-10, corrected p = 2.7640e-09 *
Yeo 2 vs Yeo 4: raw p = 1.3040e-09, corrected p = 5.4767e-09 *
Yeo 2 vs Yeo 5: raw p = 3.0511e-02, corrected p = 4.2715e-02 *
Yeo 2 vs Yeo 6: raw p = 1.5397e-13, corrected p = 1.6167e-12 *
Yeo 2 vs Yeo 7: raw p = 3.9680e-20, corrected p = 8.3328e-19 *
Yeo 3 vs Yeo 4: raw p = 5.7551e-01, corrected p = 6.0610e-01 
Yeo 3 vs Yeo 5: raw p = 9.0858e-03, corrected p = 1.5900e-02 *
Yeo 3 vs Yeo 6: raw p = 6.9168e-02, corrected p = 8.0696e-02 
Yeo 3 vs Yeo 7: raw p = 4.9689e-02, corrected p = 6.1381e-02 
Yeo 4 vs Yeo 5: raw p = 2.2087e-02, corrected p = 3.3130e-02 *
Yeo 4 vs Yeo 6: raw p = 1.6144e-02, corrected p = 2.6079e-02 *
Yeo 4 vs Yeo 7: raw p = 5.8728e-03, corrected p = 1.1212e-02 *
Yeo 5 vs Yeo 6: raw p = 1.6361e-04, corrected p = 3.8175e-04 *
Yeo 5 vs Yeo 7: raw p = 1.0492e-04, corrected p = 2.7541e-04 *
Yeo 6 vs Yeo 7: raw p = 7.6026e-01, corrected p = 7.6026e-01 

===== Within-Female Yeo Class Comparisons =====
ANOVA: F = 25.50, p = 1.3690e-25

Pairwise Yeo comparisons (t-tests, FDR-corrected):
Yeo 1 vs Yeo 2: raw p = 2.9686e-01, corrected p = 4.1561e-01 
Yeo 1 vs Yeo 3: raw p = 1.4258e-09, corrected p = 7.4854e-09 *
Yeo 1 vs Yeo 4: raw p = 2.6575e-07, corrected p = 7.3170e-07 *
Yeo 1 vs Yeo 5: raw p = 1.3984e-04, corrected p = 2.9366e-04 *
Yeo 1 vs Yeo 6: raw p = 2.7874e-07, corrected p = 7.3170e-07 *
Yeo 1 vs Yeo 7: raw p = 7.4511e-13, corrected p = 7.8236e-12 *
Yeo 2 vs Yeo 3: raw p = 1.7657e-11, corrected p = 1.2360e-10 *
Yeo 2 vs Yeo 4: raw p = 3.7801e-09, corrected p = 1.5876e-08 *
Yeo 2 vs Yeo 5: raw p = 8.6636e-06, corrected p = 2.0215e-05 *
Yeo 2 vs Yeo 6: raw p = 5.8797e-09, corrected p = 2.0579e-08 *
Yeo 2 vs Yeo 7: raw p = 4.4847e-15, corrected p = 9.4179e-14 *
Yeo 3 vs Yeo 4: raw p = 1.6915e-01, corrected p = 2.7325e-01 
Yeo 3 vs Yeo 5: raw p = 1.8779e-01, corrected p = 2.8168e-01 
Yeo 3 vs Yeo 6: raw p = 5.7752e-01, corrected p = 6.3831e-01 
Yeo 3 vs Yeo 7: raw p = 7.9682e-01, corrected p = 8.0086e-01 
Yeo 4 vs Yeo 5: raw p = 8.0086e-01, corrected p = 8.0086e-01 
Yeo 4 vs Yeo 6: raw p = 5.0259e-01, corrected p = 5.8636e-01 
Yeo 4 vs Yeo 7: raw p = 5.0619e-02, corrected p = 9.6636e-02 
Yeo 5 vs Yeo 6: raw p = 4.3814e-01, corrected p = 5.4123e-01 
Yeo 5 vs Yeo 7: raw p = 9.8429e-02, corrected p = 1.7225e-01 
Yeo 6 vs Yeo 7: raw p = 3.8371e-01, corrected p = 5.0362e-01

Note: Related to Fig.S14 and Fig.S15.

"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import warnings
import itertools
import numpy as np
import pandas as pd
import rpy2.robjects as ro
from globals import path_yeo
import matplotlib.pyplot as plt
from IPython import get_ipython
from rpy2.robjects import pandas2ri
from neuromaps.images import load_data
from rpy2.robjects.packages import importr
from scipy.stats import ttest_ind, f_oneway
from neuromaps.images import dlabel_to_gifti
from netneurotools.datasets import fetch_schaefer2018
from statsmodels.stats.multitest import multipletests
from globals import path_figures, path_info_sub, path_results
from functions import convert_cifti_to_parcellated_SchaeferTian
from functions import save_parcellated_data_in_SchaeferTian_forVis

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------

warnings.filterwarnings("ignore")
get_ipython().magic('reset -sf')

#------------------------------------------------------------------------------
# Load subject information
#------------------------------------------------------------------------------

df = pd.read_csv(path_info_sub + 'Dev_clean_data_info.csv')
num_subjects = len(df)
age = np.array(df['interview_age'])/12
sex = df.sex

#------------------------------------------------------------------------------
# Load data (vertex-wise)
#------------------------------------------------------------------------------

data_vertexwise = np.load(path_results + 'Dev_perfusion_all_vertex.npy')

data_parcellated = convert_cifti_to_parcellated_SchaeferTian(data_vertexwise.T,
                                          'cortex',
                                          'X',
                                          path_results,
                                          'rev1') # shape: (400, 627)

#------------------------------------------------------------------------------
# Stratify data by sex
#------------------------------------------------------------------------------

data_men = data_vertexwise[:, sex == 'M']
data_women = data_vertexwise[:, sex == 'F']

age_men = age[sex == 'M']
age_women = age[sex == 'F']

#------------------------------------------------------------------------------
# Activate R <-> pandas conversion
#------------------------------------------------------------------------------

pandas2ri.activate()

# Load R packages
gamlss = importr('gamlss')
base = importr('base')
stats = importr('stats')

# Load your data
sex = np.array(sex)
age = np.array(age)

#------------------------------------------------------------------------------
# Fit GAMLSS for each sex group
#------------------------------------------------------------------------------
nnodes = 400
r2_men = np.zeros(nnodes)
r2_women = np.zeros(nnodes)
mu_fits_men = {}
mu_fits_women = {}
temp = []
for i in range(nnodes):
    for label, sex_code in [('Male', 'M'), ('Female', 'F')]:
        mask = sex == sex_code
        x = age[mask]
        y = data_parcellated[i, mask]

        df_r = pd.DataFrame({'x': x, 'y': y})
        ro.globalenv['df'] = pandas2ri.py2rpy(df_r)

        try:
            ro.r('library(gamlss)')
            ro.r('model <- gamlss(y ~ pfb(x), sigma.fo = ~ pfb(x), nu.fo = ~1, data = df, family = GG())')
            mu_fitted = np.array(ro.r('fitted(model, what = "mu")'))

            ss_res = np.sum((y - mu_fitted) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared_mu = 1 - ss_res / ss_tot
            
            # Get number of observations and degrees of freedom
            n = len(y)
            df_fit = float(ro.r('model$df.fit')[0])  # Extract model degrees of freedom from R
            temp.append(df_fit)
            # Compute adjusted R²
            r_squared_adj = 1 - (1 - r_squared_mu) * (n - 1) / (n - df_fit - 1)

            if label == 'Male':
                r2_men[i] = r_squared_adj
                mu_fits_men[i] = mu_fitted
            else:
                r2_women[i] = r_squared_adj
                mu_fits_women[i] = mu_fitted

        except Exception as e:
            print(f"Error at node {i}, sex {label}: {e}")
            if label == 'Male':
                r2_men[i] = np.nan
            else:
                r2_women[i] = np.nan

# Save the R2 maps
save_parcellated_data_in_SchaeferTian_forVis(r2_men,
                                             'cortex',
                                             'S1',
                                             path_results,
                                             'r2_men')
save_parcellated_data_in_SchaeferTian_forVis(r2_women,
                                             'cortex',
                                             'S1',
                                             path_results,
                                             'r2_women')
print('------------------------------------------------------------------------')
#------------------------------------------------------------------------------
# Load schaefer parcels
#------------------------------------------------------------------------------

schaefer = fetch_schaefer2018('fslr32k')[str(nnodes) + 'Parcels7Networks']
atlas = load_data(dlabel_to_gifti(schaefer))

yeo7 = np.load(path_yeo + 'Schaefer2018_400Parcels_7Networks.npy')
yeo_labels = yeo7  + 1 # assuming shape (400,) with labels from 1 to 7

num_yeo_networks = 7

# Yeo 7 Networks Colors
yeo_colors = {
    1: [120/255, 18/255, 134/255],  # Visual Network 1
    2: [70/255, 130/255, 180/255],  # Visual Network 2
    3: [0/255, 118/255, 14/255],    # Somatomotor Network
    4: [196/255, 58/255, 250/255],  # Dorsal Attention Network
    5: [220/255, 248/255, 164/255], # Ventral Attention Network
    6: [230/255, 148/255, 34/255],  # Limbic Network
    7: [205/255, 62/255, 78/255] }  # Default Mode Network

#------------------------------------------------------------------------------
# Plot the R2 barplot
#------------------------------------------------------------------------------

def compare_yeo_within_sex(r2_array, sex_label):
    print(f"\n===== Within-{sex_label} Yeo Class Comparisons =====")

    groups = [r2_array[yeo_labels == net] for net in range(1, 8)]
    f_stat, p_anova = f_oneway(*groups)
    print(f"ANOVA: F = {f_stat:.2f}, p = {p_anova:.4e}")

    pvals = []
    comparisons = []
    for i, j in itertools.combinations(range(1, 8), 2):
        data_i = r2_array[yeo_labels == i]
        data_j = r2_array[yeo_labels == j]
        t_stat, p_val = ttest_ind(data_i, data_j, equal_var=False, nan_policy='omit')
        pvals.append(p_val)
        comparisons.append((i, j))

    reject, pvals_corrected, _, _ = multipletests(pvals, method='fdr_bh')

    sig_pairs = []
    print("\nPairwise Yeo comparisons (t-tests, FDR-corrected):")
    for (i, j), raw_p, corr_p, sig in zip(comparisons, pvals, pvals_corrected, reject):
        star = "*" if sig else ""
        if sig:
            sig_pairs.append((i, j))
        print(f"Yeo {i} vs Yeo {j}: raw p = {raw_p:.4e}, corrected p = {corr_p:.4e} {star}")
    
    return sig_pairs

sig_male = compare_yeo_within_sex(r2_men, "Male")
sig_female = compare_yeo_within_sex(r2_women, "Female")

def annotate_significant_pairs(ax, sig_pairs, bar_heights, color='black', base_offset=0.02, line_spacing=0.01):
    """
    Draw significance lines 
    """
    current_y = max(bar_heights) + base_offset
    for count, (i, j) in enumerate(sig_pairs):
        x1, x2 = i , j   # bar indices
        height = current_y + count * line_spacing
        ax.plot([x1, x1, x2, x2], [height, height + 0.005, height + 0.005, height], lw=1.2, color=color)
        ax.text((x1 + x2) / 2, height + 0.007, "*", ha='center', va='bottom', color=color, fontsize=13)

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

for ax, r2, title, sig_pairs in zip(
    axes, [r2_men, r2_women], ["Male", "Female"], [sig_male, sig_female]):

    means = []
    for net in range(1, 8):
        r2_net = r2[yeo_labels == net]
        means.append(np.nanmean(r2_net))
        x_jitter = np.random.normal(net, 0.05, size=len(r2_net))
        ax.scatter(x_jitter, r2_net, alpha=0.5, color=yeo_colors[net], s=20)

    bars = ax.bar(range(1, 8), means, color=[yeo_colors[n] for n in range(1, 8)], alpha=0.7)
    annotate_significant_pairs(ax, sig_pairs, means)

    ax.set_xticks(range(1, 8))
    ax.set_xticklabels([f"Yeo {i}" for i in range(1, 8)])
    ax.set_ylabel("R²")
    ax.set_title(f"Explained Variance (R²) by Yeo Network - {title}")

fig.tight_layout()
fig.savefig(path_figures + 'r2_yeo_significance.svg', dpi=300)

#------------------------------------------------------------------------------
# Plot results
#------------------------------------------------------------------------------

# Create plots for each sex
fig_male, ax_male = plt.subplots(figsize=(10, 6))
fig_female, ax_female = plt.subplots(figsize=(10, 6))

# Prepare age values per sex
x_vals_m = age[sex == 'M']
x_vals_f = age[sex == 'F']

# Plot all fitted lines for males
for i in range(nnodes):
    if i in mu_fits_men:
        y_fit = mu_fits_men[i]
        sorted_idx = np.argsort(x_vals_m)
        ax_male.plot(x_vals_m[sorted_idx], y_fit[sorted_idx], alpha=0.2, color='blue')

ax_male.set_title("All Fitted GAMLSS Curves - Male (400 nodes)")
ax_male.set_xlabel("Age (years)")
ax_male.set_ylabel("Fitted Perfusion")

# Plot all fitted lines for females
for i in range(nnodes):
    if i in mu_fits_women:
        y_fit = mu_fits_women[i]
        sorted_idx = np.argsort(x_vals_f)
        ax_female.plot(x_vals_f[sorted_idx], y_fit[sorted_idx], alpha=0.2, color='red')

ax_female.set_title("All Fitted GAMLSS Curves - Female (400 nodes)")
ax_female.set_xlabel("Age (years)")
ax_female.set_ylabel("Fitted Perfusion")

# Save figures
fig_male.tight_layout()
fig_female.tight_layout()

fig_male.savefig(path_figures + 'male_galss.svg', dpi=300)
fig_female.savefig(path_figures + 'female_galss.svg', dpi=300)

#------------------------------------------------------------------------------
# Plot results - color each line based on the yeo networks
#------------------------------------------------------------------------------

parcel_colors = [yeo_colors[label] for label in yeo_labels]

# Determine global y-limits from all fitted values
all_fitted = list(mu_fits_men.values()) + list(mu_fits_women.values())
y_min = min(np.min(y) for y in all_fitted)
y_max = max(np.max(y) for y in all_fitted)

# Create subplots
fig_male_grid, axes_male = plt.subplots(4, 3, figsize=(15, 10))
fig_female_grid, axes_female = plt.subplots(4, 3, figsize=(15, 10))
axes_male = axes_male.flatten()
axes_female = axes_female.flatten()

for i in range(7, 9):
    fig_male_grid.delaxes(axes_male[i])
    fig_female_grid.delaxes(axes_female[i])

x_vals_m = age[sex == 'M']
x_vals_f = age[sex == 'F']

for net in range(1, 8):
    ax_m = axes_male[net - 1]
    ax_f = axes_female[net - 1]
    color = yeo_colors[net]
    title = f"Yeo {net}"

    for i in range(nnodes):
        if yeo_labels[i] != net:
            continue
        if i in mu_fits_men:
            y_fit = mu_fits_men[i]
            sorted_idx = np.argsort(x_vals_m)
            ax_m.plot(x_vals_m[sorted_idx], y_fit[sorted_idx], alpha=0.61, color=color)
        if i in mu_fits_women:
            y_fit = mu_fits_women[i]
            sorted_idx = np.argsort(x_vals_f)
            ax_f.plot(x_vals_f[sorted_idx], y_fit[sorted_idx], alpha=0.61, color=color)

    for ax in (ax_m, ax_f):
        ax.set_title(f"{'Male' if ax is ax_m else 'Female'} - {title}")
        ax.set_xlabel("Age (years)")
        ax.set_ylabel("Fitted Perfusion")
        ax.set_ylim([y_min, y_max])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

fig_male_grid.tight_layout()
fig_female_grid.tight_layout()
fig_male_grid.savefig(path_figures + 'subplot_male_yeo.svg', dpi=300)
fig_female_grid.savefig(path_figures + 'subplot_female_yeo.svg', dpi=300)

#------------------------------------------------------------------------------
# END
