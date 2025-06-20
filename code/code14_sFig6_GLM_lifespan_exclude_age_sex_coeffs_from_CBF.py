"""

The corrected data is saved as "perfusion_clean_sex_age_2datasets_GLM.npy"
The data is concatenated across the HCP Development (HCP_D) and HCP Adult (HCP_A) datasets.

The model used is:
  1 + sex + age + age^2 + age^3 + sex*age + sex*age^2 + sex*age^3

The beta values and Bonferroni-corrected p-values are saved.

Note: Related to Fig.S6.

"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import globals
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from IPython import get_ipython
from neuromaps.images import load_data
from neuromaps.images import dlabel_to_gifti
from functions import save_as_dscalar_and_npy
from globals import path_results, path_info_sub
from netneurotools.datasets import fetch_schaefer2018
from statsmodels.stats.multitest import multipletests

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------

warnings.filterwarnings("ignore")
get_ipython().magic('reset -sf')

#------------------------------------------------------------------------------
# Load atlas and templates that might be needed!
#------------------------------------------------------------------------------

schaefer = fetch_schaefer2018('fslr32k')[f"{globals.nnodes_Schaefer}Parcels7Networks"]
atlas = load_data(dlabel_to_gifti(schaefer))

#------------------------------------------------------------------------------
# Load subject information
#------------------------------------------------------------------------------

df_age = pd.read_csv(path_info_sub + 'clean_data_info.csv')
df_age['sex'] = df_age['sex'].map({'F': 0, 'M': 1})
ageold = np.array(df_age['interview_age'])/12
sexold = df_age.sex

df_dev = pd.read_csv(path_info_sub + 'Dev_clean_data_info.csv')
df_dev['sex'] = df_dev['sex'].map({'F': 0, 'M': 1})
agedev = np.array(df_dev['interview_age'])/12
sexdev = df_dev.sex

# Concatenate age and sex across both HCP development and HCP aging
sex = np.concatenate((sexdev, sexold))
age = np.concatenate((agedev, ageold))

num_subjects = len(age)

#------------------------------------------------------------------------------
# Load data (vertex-wise) - perfusion in this case
#------------------------------------------------------------------------------

data_vertexwise_dev = np.load(path_results + 'Dev_perfusion_all_vertex.npy')
data_vertexwise_age = np.load(path_results + 'perfusion_all_vertex.npy')

# Concatenate perfusion data across both datasets
data_vertexwise = np.concatenate((data_vertexwise_dev, data_vertexwise_age), axis = 1)

#------------------------------------------------------------------------------
# Build GLM model to extract out the effect of age
#------------------------------------------------------------------------------

# Number of voxels/vertices
num_vertices = data_vertexwise.shape[0] # Assuming data is (num_subjects, num_vertices)

# Initialize arrays to store beta coefficients
betas_age = np.zeros(num_vertices)
betas_age2 = np.zeros(num_vertices)
betas_age3 = np.zeros(num_vertices)
betas_sex = np.zeros(num_vertices)
betas_agesex = np.zeros(num_vertices)
betas_age2sex = np.zeros(num_vertices)
betas_age3sex = np.zeros(num_vertices)

# Initialize arrays to store p-values
p_values_age  = np.zeros(num_vertices)
p_values_sex  = np.zeros(num_vertices)
p_values_age2  = np.zeros(num_vertices)
p_values_age3  = np.zeros(num_vertices)
p_values_agesex  = np.zeros(num_vertices)
p_values_age2sex  = np.zeros(num_vertices)
p_values_age3sex  = np.zeros(num_vertices)

# Array for residuals (to hold "corrected" data)
residuals = np.zeros((num_vertices, num_subjects))

# Loop through each voxel and fit the GLM
for i in range(num_vertices):
    y = data_vertexwise[i, :]  # Perfusion at vertex i

    # Build design matrix
    X = np.column_stack((age, sex, age*age, age*age*age, sex*age, sex*age*age, sex*age*age*age))
    X = sm.add_constant(X) # Add intercept to the model

    # Define the GLM model (using the Gaussian family for continuous data)
    model = sm.GLM(y, X, family = sm.families.Gaussian())
    results = model.fit()

    # Save beta coefficients
    betas_age[i]     = results.params[1]
    betas_sex[i]     = results.params[2]
    betas_age2[i]    = results.params[3]
    betas_age3[i]    = results.params[4]
    betas_agesex[i]  = results.params[5]
    betas_age2sex[i] = results.params[6]
    betas_age3sex[i] = results.params[7]

    # Save p-values
    p_values_age[i]     = results.pvalues[1]
    p_values_sex[i]     = results.pvalues[2]
    p_values_age2[i]    = results.pvalues[3]
    p_values_age3[i]    = results.pvalues[4]
    p_values_agesex[i]  = results.pvalues[5]
    p_values_age2sex[i] = results.pvalues[6]
    p_values_age3sex[i] = results.pvalues[7]

    # Generate "corrected" residuals:
    residuals[i,:] = results.resid_response + results.params[0]

    # Print progress
    print(i)

# Save corrected perfusion data (residuals)
np.save(path_results + 'perfusion_clean_sex_age_2datasets_GLM.npy', residuals)

#------------------------------------------------------------------------------
# Multiple-comparison correction (Bonferroni)
#------------------------------------------------------------------------------

p_values_age_corrected     = multipletests(p_values_age,     method = 'bonferroni')[1]
p_values_sex_corrected     = multipletests(p_values_sex,     method = 'bonferroni')[1]
p_values_age2_corrected    = multipletests(p_values_age2,    method = 'bonferroni')[1]
p_values_age3_corrected    = multipletests(p_values_age3,    method = 'bonferroni')[1]
p_values_agesex_corrected  = multipletests(p_values_agesex,  method = 'bonferroni')[1]
p_values_age2sex_corrected = multipletests(p_values_age2sex, method = 'bonferroni')[1]
p_values_age3sex_corrected = multipletests(p_values_age3sex, method = 'bonferroni')[1]

#------------------------------------------------------------------------------
# Save beta coefficients
#------------------------------------------------------------------------------

np.save(path_results + 'betas_age_perfusion_2datasets_GLM.npy',     betas_age)
np.save(path_results + 'betas_age2_perfusion_2datasets_GLM.npy',    betas_age2)
np.save(path_results + 'betas_age3_perfusion_2datasets_GLM.npy',    betas_age3)
np.save(path_results + 'betas_sex_perfusion_2datasets_GLM.npy',     betas_sex)
np.save(path_results + 'betas_agesex_perfusion_2datasets_GLM.npy',  betas_agesex)
np.save(path_results + 'betas_age2sex_perfusion_2datasets_GLM.npy', betas_age2sex)
np.save(path_results + 'betas_age3sex_perfusion_2datasets_GLM.npy', betas_age3sex)

#------------------------------------------------------------------------------
# Save corrected p-values (as .dscalar and .npy)
#------------------------------------------------------------------------------

save_as_dscalar_and_npy(p_values_age_corrected,     'cortex_subcortex', path_results,
                        'p_values_age_corrected_perfusion_2datasets_GLM')
save_as_dscalar_and_npy(p_values_age2_corrected,    'cortex_subcortex', path_results,
                        'p_values_age2_corrected_perfusion_2datasets_GLM')
save_as_dscalar_and_npy(p_values_age3_corrected,    'cortex_subcortex', path_results,
                        'p_values_age3_corrected_perfusion_2datasets_GLM')
save_as_dscalar_and_npy(p_values_sex_corrected,     'cortex_subcortex', path_results,
                        'p_values_sex_corrected_perfusion_2datasets_GLM')
save_as_dscalar_and_npy(p_values_agesex_corrected,  'cortex_subcortex', path_results,
                        'p_values_agesex_corrected_perfusion_2datasets_GLM')
save_as_dscalar_and_npy(p_values_age2sex_corrected, 'cortex_subcortex', path_results,
                        'p_values_age2sex_corrected_perfusion_2datasets_GLM')
save_as_dscalar_and_npy(p_values_age3sex_corrected, 'cortex_subcortex', path_results,
                        'p_values_age3sex_corrected_perfusion_2datasets_GLM')

#------------------------------------------------------------------------------
# END
