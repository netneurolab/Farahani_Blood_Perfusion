"""

PLS Cross-Validation – HCP-A- regressed out version

n = 1000 - 5 fold cross validation

p_value for first latent variable: 0.000999000999000999

    np.mean(flat_train)
    0.3643993164418383

    np.mean(flat_train_per)
    0.20491176196554925

    np.mean(flat_test)
    0.30543480001798

    np.mean(flat_test_per)
    -0.0032128891178255427

Note: Related to PLS results (Fig.S22).

"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import random
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from IPython import get_ipython
from pyls import behavioral_pls
import matplotlib.pyplot as plt
from scipy.stats import zscore, pearsonr
from sklearn.model_selection import KFold
from globals import path_results, path_figures, path_info_sub
from functions import convert_cifti_to_parcellated_SchaeferTian

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------

warnings.filterwarnings("ignore")
get_ipython().magic('reset -sf')
random.seed(0)

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

#------------------------------------------------------------------------------
#                            PLS analysis - main
#------------------------------------------------------------------------------

data = data.T
data = data[mask, :]
X = zscore(data, axis = 0)
Y = zscore(behaviour_data_array, axis = 0)

n_splits = 5
lv = 0
nperm = 1000

def cv_cal(X, Y):
    corr_test = np.zeros((n_splits, nperm))
    corr_train = np.zeros((n_splits, nperm))

    for iter_ind in range(nperm):
        kf = KFold(n_splits = n_splits, shuffle = True)
        c = 0
        for train_index, test_index in kf.split(X):

            Xtrain, Xtest = X[train_index], X[test_index]
            Ytrain, Ytest = Y[train_index], Y[test_index]

            train_result = behavioral_pls(Xtrain,
                                          Ytrain,
                                          n_boot = 0,
                                          n_perm = 0,
                                          test_split = 0,
                                          seed = 10)
            corr_train[c, iter_ind], _ = pearsonr(train_result['x_scores'][:, lv],
                                            train_result['y_scores'][:, lv])

            # project weights, correlate predicted scores in the test set
            corr_test[c, iter_ind], _ = pearsonr(Xtest @ train_result['x_weights'][:, lv],
                                   Ytest @ train_result['y_weights'][:, lv])
            c = c + 1
    return(corr_train, corr_test)

corr_train, corr_test = cv_cal(X, Y)

#------------------------------------------------------------------------------
# Permutation step
#------------------------------------------------------------------------------

def single_cv_cal(X, Y):
    corr_test = np.zeros((n_splits, 1))
    corr_train = np.zeros((n_splits, 1))
    kf = KFold(n_splits = n_splits, shuffle = True)
    c = 0
    for train_index, test_index in kf.split(X):
        Xtrain, Xtest = X[train_index], X[test_index]
        Ytrain, Ytest = Y[train_index], Y[test_index]

        train_result = behavioral_pls(Xtrain,
                                      Ytrain,
                                      n_boot = 0,
                                      n_perm = 0,
                                      test_split = 0,
                                      seed = 10)
        corr_train[c, 0], _ = pearsonr(train_result['x_scores'][:, lv],
                                        train_result['y_scores'][:, lv])

        # project weights, correlate predicted scores in the test set
        corr_test[c, 0], _ = pearsonr(Xtest @ train_result['x_weights'][:, lv],
                               Ytest @ train_result['y_weights'][:, lv])
        c = c + 1
    return(corr_train.flatten(), corr_test.flatten())

per_train_corr = np.zeros((n_splits, nperm))
per_test_corr = np.zeros((n_splits, nperm))

num_subjects = len(X)
perms_y = np.zeros((num_subjects, nperm))

for perm_ind in range(nperm):
    perms_y[:, perm_ind] = np.random.permutation(range(0, num_subjects))

for perm_ind in range(nperm):
    tempy = perms_y[:, perm_ind].astype(int)
    Y_permuted = Y[tempy]
    per_train_corr[:, perm_ind], per_test_corr[:, perm_ind] = single_cv_cal(X, Y_permuted)
    print(perm_ind)

# VISUALIZATION ---------------------------------------------------------------

flat_train = corr_train[0, :].flatten()
flat_test  = corr_test[0, :].flatten()

flat_train_per = per_train_corr[0,:].flatten()
flat_test_per  = per_test_corr[0,:].flatten()

p_val = (1 + np.count_nonzero(((flat_test_per - np.mean(flat_test_per)))
                                > ((np.mean(flat_test) - np.mean(flat_test_per))))) / (nperm + 1)

print('pval is (for lv = ' + str(lv) + '): ' + str(p_val))

p_val = (1 + np.count_nonzero((flat_test_per > (np.mean(flat_test))))) / (nperm + 1)

print('pval is (for lv = ' + str(lv) + '): ' + str(p_val))

combined_train_test = [flat_train, flat_train_per, flat_test, flat_test_per]

plt.boxplot(combined_train_test)
plt.xticks([1, 2, 3, 4], ['train', 'train permute', 'test', 'test permute'])
for spine in plt.gca().spines.values():
    spine.set_visible(False)
ax = plt.gca()
y_ticks = np.linspace(-0.5, 1, num = 5)
ax.set_yticks(y_ticks)
plt.savefig(path_figures + 'PLS_cross_validation_regressout_' + str(lv) +'.svg',
        bbox_inches = 'tight',
        dpi = 300,
        transparent = True)
plt.show()

#------------------------------------------------------------------------------
# END
