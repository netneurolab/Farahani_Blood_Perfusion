"""

PLS  - use HCP-aging dataset - male/female

    n = 1000
    MALE

    Singular values for latent variable 0: 0.8196
    x-score and y-score Spearman correlation for latent variable 0:           0.3953
    x-score and y-score Pearson correlation for latent variable 0:           0.4122

    FEMALE:
    Singular values for latent variable 0: 0.9065
    x-score and y-score Spearman correlation for latent variable 0:           0.3656
    x-score and y-score Pearson correlation for latent variable 0:           0.4080

    p-value:
        first latent variable - MALE : 9.99000999e-04
        first latent variable - FEMALE : 9.99000999e-04

MALE, barplot, error bar details:

loading values:    [-0.2470092 , -0.2417536 , -0.21754923, -0.19110124, -0.16905557,
                    -0.16221884, -0.13377514, -0.12371159, -0.10651662, -0.05980868,
                    -0.05146695, -0.03152842, -0.01617707, -0.00866467, -0.00433129,
                    -0.00368497,  0.00102053,  0.01376548,  0.01778433,  0.02174508,
                     0.03248267,  0.04243911,  0.05450708,  0.06175069,  0.06622017,
                     0.07249149,  0.0831067 ,  0.17910937]

error bars:    [0.11174309*, 0.10305604*, 0.12002132*, 0.12222203*, 0.1280611* ,
               0.08668099*, 0.14481017, 0.12500112, 0.09309711*, 0.15868685,
               0.10529306, 0.11988576, 0.12662988, 0.13366919, 0.14057211,
               0.14259023, 0.11726487, 0.13588004, 0.12438001, 0.11928275,
               0.12154086, 0.14146439, 0.12668794, 0.12032664, 0.13833468,
               0.09890172, 0.11248808, 0.10499827*])

FEMALE, barplot, error bar details:
loading values:    [-0.29817967, -0.26225076, -0.20969434, -0.19805919, -0.19135129,
                   -0.1817658 , -0.16305902, -0.14779925, -0.14510127, -0.14448076,
                   -0.14184076, -0.11278913, -0.11025103, -0.10713334, -0.10121159,
                   -0.07475111, -0.07146591, -0.0705066 , -0.05183112, -0.04970586,
                   -0.01563645, -0.00244664,  0.00232045,  0.04284898,  0.05157618,
                    0.08464676,  0.09455606,  0.12549777]

error bars:    [0.10513158, 0.10388393, 0.10696465, 0.11369009, 0.1148512 ,
                0.1159748 , 0.11820949, 0.16026991, 0.11054458, 0.12085765,
                0.12105411, 0.10751166, 0.13066482, 0.12283831, 0.12335432,
                0.11569483, 0.09479216, 0.10513414, 0.11001701, 0.10782983,
                0.10056464, 0.11165749, 0.1230243 , 0.09966599, 0.11592775,
                0.09091523, 0.10252471, 0.09927724]

Note: Related to Fig.5d,e.

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
mask_sex = (sex == 0) # 0 is female and 1 is male

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

spins = np.zeros((num_subjects, nspins))
for spin_ind in range(nspins):
    spins[:,spin_ind] = np.random.permutation(range(0, num_subjects))

spins = spins.astype(int)
np.save(path_results + 'spin_PLS_aging_main_female.npy', spins)

# Use the already created spin (for the main PLS)
spins = np.load(path_results + 'spin_PLS_aging_main_female.npy')

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
    plt.savefig(path_figures  + 'scatter_PLS_female.svg', format = 'svg')
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
    plt.savefig(path_figures + title + '_female.svg', format = 'svg')
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
# Loading plots
#------------------------------------------------------------------------------

cmap = plt.get_cmap('coolwarm')
colors = cmap(np.linspace(0, 1, 100))
def plot_loading_bar(lv, pls_result, combined_columns, vmin_val, vmax_val):
    """
    Create a horizontal bar plot of loadings, ordered by magnitude, and mark significance.
    Significance is determined based on the confidence interval crossing zero.
    """
    err = (pls_result["bootres"]["y_loadings_ci"][:, lv, 1] -
           pls_result["bootres"]["y_loadings_ci"][:, lv, 0]) / 2
    values = pls_result.y_loadings[:, lv]

    # Determine significance: if CI crosses zero, it's non-significant
    significance = (pls_result["bootres"]["y_loadings_ci"][:, lv, 1] * \
                    pls_result["bootres"]["y_loadings_ci"][:, lv, 0]) > 0

    # Sort values, errors, and significance by loading magnitude
    sorted_indices = np.argsort(values)
    values_sorted = values[sorted_indices]
    err_sorted = err[sorted_indices]
    labels_sorted = np.array(combined_columns)[sorted_indices]
    significance_sorted = significance[sorted_indices]

    # Plot the loadings
    plt.figure(figsize = (10, 10))
    bars = plt.barh(labels_sorted,
                    values_sorted,
                    xerr = err_sorted,
                    color=[colors[90] if sig else 'gray' for sig in significance_sorted])

    plt.xlabel('x-loading')
    plt.ylabel('Behavioral Measure')
    plt.title(f'Latent Variable {lv + 1} Loadings')

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
    plt.savefig(path_figures + 'PLS_bars_female.svg',
                format = 'svg')
    plt.show()

# Example usage for latent variable 0
plot_loading_bar(0, pls_result, names, -0.5, 0.5)

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
    np.save(path_results + 'loadings_cortex_PLS_lv' + str(lv) + '_female.npy', loadings_cortex)
    
    # Brain score visualization
    save_parcellated_data_in_SchaeferTian_forVis(loadings_cortex,
                            'cortex',
                            'X',
                            path_results,
                            'loadings_cortex_PLS_lv' + str(lv) + '_female')

#------------------------------------------------------------------------------
# END
