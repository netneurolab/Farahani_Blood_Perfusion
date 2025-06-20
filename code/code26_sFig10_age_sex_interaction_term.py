"""

In response to rev1:
    Blood perfusion across development section
    the authors need to report main effects vs interactions for age, sex and age x sex.
    The way the section is described right now suggests an age x sex interaction,
    whereby up to age 15 or so there are no sex differences, which then emerge
    and get stronger with age. That makes sense; however this conclusion should
    be supported by appropriate stats (ie interaction testing).

NOTE: Related to Fig.S10.

"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from globals import path_results, path_info_sub, path_figures

#------------------------------------------------------------------------------
# Load subject information
#------------------------------------------------------------------------------

df = pd.read_csv(path_info_sub + 'Dev_clean_data_info.csv')
num_subjects = len(df)
age = np.array(df['interview_age'])/12
df['sex'] = df['sex'].map({'F': 0, 'M': 1})
sex = np.array(df['sex'])

#------------------------------------------------------------------------------
# Load data (vertex-wise) - perfusion (development only)
#------------------------------------------------------------------------------

data_vertexwise = np.load(path_results + 'Dev_perfusion_all_vertex.npy')
y = np.mean(data_vertexwise, axis = 0)

#------------------------------------------------------------------------------
# Build GLM model to extract out the effect of age and sex
#------------------------------------------------------------------------------

# Build design matrix with age and sex as predictors
interaction = age*sex 
X = np.column_stack((age, sex, interaction)) #coding:0/1;0 = female;Effects relative to females
X = sm.add_constant(X)  # Add intercept to the model

# Define the GLM model (using the Gaussian family for continuous data)
model = sm.GLM(y, X, family = sm.families.Gaussian())
results = model.fit()

# Store the beta coefficient and p-value for age
betas_age = results.params[1] # Beta for age
p_values_age = results.pvalues[1] # P-value for age
# Store the beta coefficient and p-value for age
betas_sex = results.params[2] # Beta for sex
p_values_sex = results.pvalues[2] # P-value for age
betas_interaction = results.params[3]
p_values_interaction= results.pvalues[3]

print(f"Beta (age): {betas_age:.4f}, p-value: {p_values_age:.4e}")
print(f"Beta (sex): {betas_sex:.4f}, p-value: {p_values_sex:.4e}")
print(f"Beta (interaction): {betas_interaction:.4f}, p-value: {p_values_interaction:.4e}")

#------------------------------------------------------------------------------
# Visualize the differences
#------------------------------------------------------------------------------

# Get predicted values from model
y_pred = results.fittedvalues

plt.figure(figsize=(7, 5))

age_f = age[sex == 0]
age_m = age[sex == 1]
y_f = y[sex == 0]
y_m = y[sex == 1]

cmap = plt.get_cmap('coolwarm')
female_color = cmap(90 / 100)  # light blue
male_color = cmap(10 / 100)    # light red

# Plot raw data
plt.scatter(age_f, y_f, color=female_color, label='Female', alpha=0.6)
plt.scatter(age_m, y_m, color=male_color, label='Male', alpha=0.6)

age_range = np.linspace(age.min(), age.max(), 100)

# Female: sex = 0 → interaction = 0
X_f = sm.add_constant(np.column_stack((age_range, np.zeros_like(age_range), np.zeros_like(age_range))))
y_fitted = results.predict(X_f)
plt.plot(age_range, y_fitted, color= female_color, linewidth=2)

# Male: sex = 1 → interaction = age_range
X_m = np.column_stack((
    age_range,
    np.ones_like(age_range),
    age_range
))
X_m = sm.add_constant(X_m, has_constant='add')
y_fitted_m = results.predict(X_m)

# Plot the results
plt.plot(age_range, y_fitted_m, color=male_color, linewidth=2)
plt.xlabel("Age (years)")
plt.ylabel("Mean Perfusion")
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.legend()
plt.tight_layout()
plt.savefig(path_figures + 'development_sex_age_interaction.svg', format='svg')
plt.show()
#------------------------------------------------------------------------------
# END
