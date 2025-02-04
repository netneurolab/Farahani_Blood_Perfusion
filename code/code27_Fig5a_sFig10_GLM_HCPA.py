"""

Build a GLM model to quantify the effects of age and sex in the HCP aging dataset

The cleaned data is saved as:
    perfusion_clean_sex_age.npy

We also save the beta coefficients and p-values for age and sex as:
    Aging_betas_age_perfusion.npy and Aging_betas_age_perfusion.dscalar.nii
    Aging_betas_sex_perfusion.npy and Aging_betas_sex_perfusion.dscalar.nii

    Aging_p_values_age_corrected_perfusion.npy and Aging_p_values_age_corrected_perfusion.dscalar.nii
    Aging_p_values_sex_corrected_perfusion.npy and Aging_p_values_sex_corrected_perfusion.dscalar.nii

Note: Related to Fig.5a & Fig.S10.

"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from IPython import get_ipython
from functions import save_as_dscalar_and_npy
from globals import path_results, path_info_sub
from statsmodels.stats.multitest import multipletests

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
age = np.array(df['interview_age'])/12
df['sex'] = df['sex'].map({'F': 0, 'M': 1})
sex = np.array(df['sex'])

#------------------------------------------------------------------------------
# Load data (vertex-wise) - perfusion (aging only)
#------------------------------------------------------------------------------

data_vertexwise = np.load(path_results + 'perfusion_all_vertex.npy')

#------------------------------------------------------------------------------
# Build GLM model to extract out the effect of age and sex - linear effects
#------------------------------------------------------------------------------

# Number of voxels/vertices
num_vertices = data_vertexwise.shape[0] # Assuming data is (num_subjects, num_vertices)

# Initialize arrays to store results
betas_age = np.zeros(num_vertices)
betas_sex = np.zeros(num_vertices)

p_values_age  = np.zeros(num_vertices)
p_values_sex  = np.zeros(num_vertices)

residuals = np.zeros((num_vertices, len(df)))

# Loop through each voxel and fit the GLM
for i in range(num_vertices):
    y = data_vertexwise[i, :]  # Perfusion at vertex i
    # Build design matrix with age and sex as predictors
    X = np.column_stack((age, sex))
    X = sm.add_constant(X)  # Add intercept to the model

    # Define the GLM model (using the Gaussian family for continuous data)
    model = sm.GLM(y, X, family = sm.families.Gaussian())
    results = model.fit()

    # Store the beta coefficient and p-value for age
    betas_age[i] = results.params[1]  # Beta for age
    p_values_age[i] = results.pvalues[1]  # P-value for age
    # Store the beta coefficient and p-value for age
    betas_sex[i] = results.params[2]  # Beta for sex
    p_values_sex[i] = results.pvalues[2]  # P-value for age

    residuals[i,:] = results.resid_response  #+ results.params[0]
    print(i)

np.save(path_results + 'perfusion_clean_sex_age.npy', residuals)
p_values_sex_corrected = [multipletests(p_values_sex[:], method = 'bonferroni')][0][1]
p_values_age_corrected = [multipletests(p_values_age[:], method = 'bonferroni')][0][1]

#------------------------------------------------------------------------------
# Save the results as a dscalar file and also as a npy file
#------------------------------------------------------------------------------

save_as_dscalar_and_npy(betas_age,
                        'cortex_subcortex',
                        path_results,
                        'Aging_betas_age_perfusion')

save_as_dscalar_and_npy(betas_sex,
                        'cortex_subcortex',
                        path_results,
                        'Aging_betas_sex_perfusion')

save_as_dscalar_and_npy(p_values_sex_corrected,
                        'cortex_subcortex',
                        path_results,
                        'Aging_p_values_sex_corrected_perfusion')

save_as_dscalar_and_npy(p_values_age_corrected,
                        'cortex_subcortex',
                        path_results,
                        'Aging_p_values_age_corrected_perfusion')

#------------------------------------------------------------------------------
# END