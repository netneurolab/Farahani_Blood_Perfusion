"""

Rev1 - response to:
    "For future PLS - if the authors are using pls bootstrap for robustness of weights,
    its worth reporting the weights as Z-scores to assess stability 
    (see Morgan et al 2019 PNAS, Zhukovsky et al 2022 PNAS for code).
    Is it a PLS correlation or a PLS-regression? PLSC/PLSR makes this clear to the reader."

Note: Related to Fig.S21.

"""

#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from IPython import get_ipython
from pyls import behavioral_pls
import matplotlib.pyplot as plt
from scipy.stats import zscore, pearsonr, spearmanr
from globals import path_results, path_figures, path_info_sub
from functions import convert_cifti_to_parcellated_SchaeferTian
from functions import save_parcellated_data_in_SchaeferTian_forVis

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------

warnings.filterwarnings("ignore")
get_ipython().magic('reset -sf')

#------------------------------------------------------------------------------
# Load data (vertex-wise) - perfusion data from the aging dataset
#------------------------------------------------------------------------------

data_vertexwise = np.load(path_results +'perfusion_all_vertex.npy') # perfusion data

num_vertices = data_vertexwise.shape[0] # Assuming data is (num_subjects, num_vertices)
num_subjects = data_vertexwise.shape[1]

# Parcellate data based on Schaefer-400 parcellation
data = convert_cifti_to_parcellated_SchaeferTian(data_vertexwise.T,
                                                 'cortex',
                                                 'S1',
                                                 path_results,
                                                 'data_PLS_aging_main')

#------------------------------------------------------------------------------
# Load blood test + demographics data, clean them, and add BMI as one of the measures here
#------------------------------------------------------------------------------

finalDF = pd.read_csv(path_results + 'finalDF.csv')
finalDF['bmi'] = (finalDF["weight_std (Weight - Standard Unit)"] * 0.453592)/ \
    (finalDF["vtl007 (Height in inches)"]*(finalDF["vtl007 (Height in inches)"])*0.0254*0.0254)

# Filter the column to exclude values equal to 99999/0999
valid_values = finalDF.loc[(finalDF['friedewald_ldl (Friedewald LDL Cholesterol: )'] != 99999) &
                           (finalDF['friedewald_ldl (Friedewald LDL Cholesterol: )'] != 9999),
                           'friedewald_ldl (Friedewald LDL Cholesterol: )']

# Calculate the median of the valid values
median_value = valid_values.median()

# Replace the 99999/9999 values with the calculated median
finalDF.loc[(finalDF['friedewald_ldl (Friedewald LDL Cholesterol: )'] == 99999) |
            (finalDF['friedewald_ldl (Friedewald LDL Cholesterol: )'] == 9999),
            'friedewald_ldl (Friedewald LDL Cholesterol: )'] = median_value

# Filter the column to exclude values equal to 9999
valid_values_2 = finalDF.loc[finalDF['a1crs (HbA1c Results)'] != 9999,
                           'a1crs (HbA1c Results)']

# Calculate the median of the valid values
median_value_2 = valid_values_2.median()

# Replace the 9999 values with the calculated median
finalDF.loc[finalDF['a1crs (HbA1c Results)'] == 9999,
            'a1crs (HbA1c Results)'] = median_value_2


finalDF = finalDF[
    [   "interview_age",
        "lh (LH blood test result)",
        "fsh (FSH blood test result)",
        "festrs (Hormonal Measures Female Estradiol Results)",
        "rsptest_no (Blood value - Testosterone (ng/dL))",
        "ls_alt (16. ALT(SGPT)(U/L))",
        "rsptc_no (Blood value - Total Cholesterol (mg/dL))",
        "rsptrig_no (Blood value - Triglycerides (mg/dL))",
        "ls_ureanitrogen (10. Urea Nitrogen(mg/dL))",
        "ls_totprotein (11. Total Protein(g/dL))",
        "ls_co2 (7. CO2 Content(mmol/L))",
        "ls_calcium (17. Calcium(mg/dL))",
        "ls_bilirubin (13. Bilirubin, Total(mg/dL))",
        "ls_albumin (12. Albumin(g/dL))",
        "laba5 (Hdl Cholesterol (Mg/Dl))",
        "bp1_alk (Liver ALK Phos)",
        "a1crs (HbA1c Results)",
        "insomm (Insulin: Comments)",
        "friedewald_ldl (Friedewald LDL Cholesterol: )",
        "vitdlev (25- vitamin D level (ng/mL))",
        "ls_ast (15. AST(SGOT)(U/L))",
        "glucose (Glucose)",
        "chloride (Chloride)",
        "creatinine (Creatinine)",
        "potassium (Potassium)",
        "sodium (Sodium)",
        "MAP (nan)",
        "bmi"
    ]
]

# Add sex as a variable of interest
df = pd.read_csv(path_info_sub + 'clean_data_info.csv')
num_subjects = len(df)
age = np.array(df['interview_age'])/12
df['sex'] = df['sex'].map({'F': 0, 'M': 1})
sex = np.array(df['sex'])

beh_data = finalDF
name_write =["age",
             "lh",
             "fsh",
            "festrs",
            "Testosterone",
            "ALT",
            "Cholesterol",
            "Triglycerides",
            "Urea",
            "Protein",
            "CO2",
            "Calcium",
            "Bilirubin",
            "Albumin", 
            "HDL",
            "Liver_ALK",
            "HbA1c",
            "Insulin",
            "LDL",
            "vitamin_D",
            "AST",
            "Glucose",
            "Chloride",
            "Creatinine",
            "Potassium",
            "Sodium",
            "MAP",
            "BMI"]


# Count the number of NaN values per subject (row)
nan_counts = beh_data.isna().sum(axis = 1)

# Create a mask for subjects that have 20 or fewer NaN values
mask_nan = nan_counts <= 20
# Create a mask for male/female subjects
mask_sex = (sex == 1) # 0 is female and 1 is male

# Combine the two masks
# Only subjects who pass both filters (<=20 NaNs and male/female) remain.
final_mask = mask_nan & mask_sex

# Filter the data frames/arrays using the mask
beh_data = beh_data[final_mask] # Keep only those subjects in beh_data

# Mode imputation - missing values
beh_data.fillna(beh_data.mode().iloc[0], inplace = True)
behaviour_data_array = (np.array(beh_data))
names = beh_data.columns

#------------------------------------------------------------------------------
#                            PLS analysis - main
#------------------------------------------------------------------------------

# only inlcude data of male/female
data = data[:, final_mask]
data = data.T
X = zscore(data, axis = 0)
Y = zscore(behaviour_data_array, axis = 0)

nspins = 1000
num_subjects = (sum(final_mask))
'''
spins = np.zeros((num_subjects, nspins))
for spin_ind in range(nspins):
    spins[:,spin_ind] = np.random.permutation(range(0, num_subjects))

spins = spins.astype(int)
np.save(path_results + 'spin_PLS_aging_main_male.npy', spins)
'''
# Use the already created spin (for the main PLS)
spins = np.load(path_results + 'spin_PLS_aging_main_male.npy')

pls_result = behavioral_pls(X,
                            Y,
                            n_boot = nspins,
                            n_perm = nspins,
                            permsamples = spins,
                            test_split = 0,
                            seed = 0)

#------------------------------------------------------------------------------
# plot scatter plot (scores)
#------------------------------------------------------------------------------

def plot_scores_and_correlations_unicolor(lv,
                                          pls_result,
                                          title,
                                          clinical_scores,
                                          path_fig,
                                          column_name):

    plt.figure(figsize = (5,5))
    plt.scatter(range(1, len(pls_result.varexp) + 1),
                pls_result.varexp,
                color = 'gray')
    plt.savefig(path_figures  + 'scatter_PLS_male.svg', format = 'svg')
    plt.title(title)
    plt.xlabel('Latent variables')
    plt.ylabel('Variance Explained')

    # Calculate and print singular values
    singvals = pls_result["singvals"] ** 2 / np.sum(pls_result["singvals"] ** 2)
    print(f'Singular values for latent variable {lv}: {singvals[lv]:.4f}')

    # Plot score correlation
    plt.figure(figsize = (5, 5))
    plt.title(title)

    sns.regplot(x = pls_result['x_scores'][:, lv],
                y = pls_result['y_scores'][:, lv],
                scatter = False)
    sns.scatterplot(x = pls_result['x_scores'][:, lv],
                    y = pls_result['y_scores'][:, lv],
                    c = clinical_scores,
                    s = 30,
                    cmap = 'coolwarm',
                    vmin = 0,
                    vmax = 1,
                    edgecolor='black',
                    linewidth = 0.5)

    plt.xlabel('X scores')
    plt.ylabel('Y scores')
    plt.tight_layout()

    # Calculate and print score correlations
    score_correlation_spearmanr = spearmanr(pls_result['x_scores'][:, lv],
                                            pls_result['y_scores'][:, lv])
    score_correlation_pearsonr = pearsonr(pls_result['x_scores'][:, lv],
                                          pls_result['y_scores'][:, lv])

    print(f'x-score and y-score Spearman correlation for latent variable {lv}: \
          {score_correlation_spearmanr.correlation:.4f}')
    print(f'x-score and y-score Pearson correlation for latent variable {lv}: \
          {score_correlation_pearsonr[0]:.4f}')

for behavior_ind in range(np.size(Y, axis = 1)):
    for lv in range(1): # Plot Scores and variance explained figures
        title = f'Latent Variable {lv + 1}'
        column_name = (names[behavior_ind])
        colors = (Y[:,behavior_ind] - min(Y[:,behavior_ind])) / (max(Y[:,behavior_ind]) - min(Y[:,behavior_ind]))
        plot_scores_and_correlations_unicolor(lv,
                                              pls_result,
                                              name_write[behavior_ind],
                                              colors,
                                              path_results,
                                              names)

#------------------------------------------------------------------------------
# Flip X and Y in PLS to get x-loading confidence intervals
#------------------------------------------------------------------------------

xload = behavioral_pls(Y,
                       X,
                       n_boot = 1000,
                       n_perm = 0,
                       test_split = 0,
                       seed = 0)
for lv in range(1):
    loadings_cortex = xload.y_loadings[:, lv]

    # Brain score visualization
    save_parcellated_data_in_SchaeferTian_forVis(loadings_cortex,
                            'cortex',
                            'X',
                            path_results,
                            'rev1_loadings_cortex_PLS_lv' + str(lv) + '_male')

#------------------------------------------------------------------------------
# z-scored - bootstrap plots
#------------------------------------------------------------------------------

cmap = plt.get_cmap('coolwarm')
colors = cmap(np.linspace(0, 1, 100))

def plot_bar(lv, pls_result, combined_columns, vmin_val, vmax_val):

    values = xload["bootres"]["x_weights_normed"][:,lv]

    # Determine significance: if CI crosses zero, it's non-significant
    significance = (xload["bootres"]["y_loadings_ci"][:, lv, 1] * \
                    xload["bootres"]["y_loadings_ci"][:, lv, 0]) > 0

    # Sort values
    sorted_indices = np.argsort(values)
    values_sorted = values[sorted_indices]

    labels_sorted = np.array(combined_columns)[sorted_indices]
    significance_sorted = significance[sorted_indices]

    # Define colors based on z-threshold
    colors_bar = [colors[90]if (v > 1.96 or v < -1.96) else 'gray' for v in values_sorted]
    plt.figure(figsize=(10, 10))
    bars = plt.barh(labels_sorted,
                    values_sorted,
                    color=colors_bar)
    plt.xlabel('x-loading')
    plt.ylabel('Behavioral Measure')
    plt.title(f'Latent Variable {lv + 1} Loadings')
    plt.axvline(x=1.96, color='black', linestyle='--', linewidth=1)
    plt.axvline(x=-1.96, color='black', linestyle='--', linewidth=1)
    plt.axvline(x=3, color='green', linestyle='--', linewidth=1)
    plt.axvline(x=-3, color='green', linestyle='--', linewidth=1)

    # Highlight significant loadings by making them bold
    for bar, sig in zip(bars, significance_sorted):
        if sig:
            bar.set_linewidth(1.5)
            bar.set_edgecolor('black')
    ax = plt.gca()
    x_ticks = np.linspace(vmin_val, vmax_val, num = 5)
    ax.set_xticks(x_ticks)
    plt.xticks(rotation = 90)
    plt.tight_layout()
    plt.savefig(path_figures + 'rev1_PLS_bars_male.svg',
                format = 'svg')
    plt.show()

# Example usage for latent variable 0
plot_bar(0, xload, names, -6.5, 6.5)
#------------------------------------------------------------------------------
# END
