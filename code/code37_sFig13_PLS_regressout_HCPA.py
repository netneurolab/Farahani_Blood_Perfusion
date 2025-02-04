"""

PLS – HCP-A; Linear Effect of Age and Sex Removed from Data

n = 1000

Singular values for latent variable 0: 0.9430
x-score and y-score Spearman correlation for latent variable 0:           0.3391
x-score and y-score Pearson correlation for latent variable 0:           0.3563

p-value:
    first latent variable: 9.99000999e-04

Bar plot loadings:
            [-0.2317821 , -0.2156386 , -0.21071935, -0.19158466, -0.1549183 ,
           -0.1374659 , -0.12511233, -0.1028036 , -0.09143277, -0.07309417,
           -0.06573724, -0.05901464, -0.05430012, -0.03433263, -0.02438459,
           -0.01795641, -0.00543813,  0.0030331 ,  0.01560143,  0.01631853,
            0.02536967,  0.03436234,  0.03880921,  0.04239076,  0.04413487,
            0.05456101,  0.05725282,  0.06618098,  0.17035796]

Bar plot error bars:
           [0.08706209*, 0.08029159*, 0.07884313*, 0.06897917*, 0.06530732*,
           0.09335404*, 0.07601905*, 0.07459415*, 0.08687661*, 0.07644331,
           0.08732857, 0.08035691, 0.08043999, 0.07501062, 0.07920301,
           0.07838501, 0.07975476, 0.08210237, 0.07843605, 0.07344022,
           0.07719088, 0.08483396, 0.08713312, 0.07861422, 0.08639009,
           0.07788878, 0.08097095, 0.07159436, 0.07421824*]

Note: Related to Fig.S13.

"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
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
# Load data (vertex-wise) - perfusion - corrected for linear effect of age and sex
#------------------------------------------------------------------------------

residuals_data = np.load(path_results + 'perfusion_clean_sex_age.npy') # cleaned perfusion data
data = convert_cifti_to_parcellated_SchaeferTian(residuals_data.T,
                                                 'cortex',
                                                 'S1',
                                                 path_results,
                                                 'residuals')

num_vertices = residuals_data.shape[0] # Assuming data is (num_subjects, num_vertices)
num_subjects = residuals_data.shape[1]

# Load subject information here
df = pd.read_csv(path_info_sub + 'clean_data_info.csv')
num_subjects = len(df)
age = np.array(df['interview_age'])/12
df['sex'] = df['sex'].map({'F': 0, 'M': 1})
sex = np.array(df['sex'])

#------------------------------------------------------------------------------
# Load behavioral data and perfom some cleaning
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
    [
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

beh_data = finalDF
name_write =[
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
            "BMI",
            "age",
            "sex"]

# Count the number of NaN values per subject (row)
nan_counts = beh_data.isna().sum(axis = 1)

# Create a mask for subjects that have 20 or fewer NaN values
mask = nan_counts <= 20

# Filter the data frames/arrays using the mask
beh_data = beh_data[mask]  # Keep only those subjects in bex_data
age = age[mask]
sex = sex[mask]

# Mode imputation - missing values
beh_data.fillna(beh_data.mode().iloc[0], inplace = True)
behaviour_data_array = (np.array(beh_data))
names = name_write

# Regress out age and sex from behavioral data ________________________________

residuals_behavioral = np.zeros((len(behaviour_data_array.T), (sum(mask))))
for i in range(0, len(behaviour_data_array.T)):
    y = behaviour_data_array[:, i]
    X = np.column_stack((age, sex))
    X = sm.add_constant(X)
    model = sm.GLM(y, X, family = sm.families.Gaussian())
    results = model.fit()
    residuals_behavioral[i, :] =  results.resid_response
    print(i)

behaviour_data_array = residuals_behavioral.T

# Add sex as a variable of interest
my_age = age[:, np.newaxis]  # Reshape to (597, 1)
behaviour_data_array = np.concatenate((behaviour_data_array, my_age), axis = 1)

my_sex = sex[:, np.newaxis]  # Reshape to (597, 1)
behaviour_data_array = np.concatenate((behaviour_data_array, my_sex), axis = 1)

# Assume 0 = Female and 1 = Male
female_mask = sex == 0
male_mask = sex == 1

for i in range(len(beh_data.columns)):
    plt.figure()
    plt.scatter(age[female_mask],
                behaviour_data_array[female_mask, i],
                color='red',
                label='Female')
    plt.scatter(age[male_mask],
                behaviour_data_array[male_mask, i],
                color='blue',
                label='Male')
    plt.title(name_write[i])
    plt.xlabel('Interview Age')
    plt.ylabel(name_write[i])
    plt.legend()
    plt.show()

#------------------------------------------------------------------------------
#                            PLS analysis - main
#------------------------------------------------------------------------------

data = data.T
data = data[mask, :]
X = zscore(data, axis = 0)
Y = zscore(behaviour_data_array, axis = 0)

nspins = 1000
num_subjects = len(X)

spins = np.zeros((num_subjects, nspins))
for spin_ind in range(nspins):
    spins[:,spin_ind] = np.random.permutation(range(0, num_subjects))
spins = spins.astype(int)
np.save(path_results + 'spin_PLS_aging_main.npy', spins)

# Use the already created spin ( for the main PLS )
spins = np.load(path_results + 'spin_PLS_aging_main.npy')

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
    plt.savefig(path_figures  + 'scatter_PLS_regressout.svg', format = 'svg')
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
    plt.savefig(path_figures + title + '_regressout.svg', format = 'svg')
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
    for bar, sig in zip(bars, significance_sorted):
        if sig:
            bar.set_linewidth(1.5)
            bar.set_edgecolor('black')
    ax = plt.gca()
    x_ticks = np.linspace(vmin_val, vmax_val, num = 5)
    ax.set_xticks(x_ticks)
    plt.xticks(rotation = 90)
    plt.tight_layout()
    plt.savefig(path_figures + 'PLS_bars_regressout.svg', format = 'svg')
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
    np.save(path_results + 'loadings_cortex_PLS_regressout_lv' + str(lv) + '.npy', loadings_cortex)
    
    # Brain score visualization
    save_parcellated_data_in_SchaeferTian_forVis(loadings_cortex,
                            'cortex',
                            'X',
                            path_results,
                            'loadings_cortex_PLS_regressout_lv' + str(lv))

#------------------------------------------------------------------------------
# END