"""

Show the panels and physiological measures of HCP-aging cohort

NOTE: Related to Fig.S20.

"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import warnings
import numpy as np
import pandas as pd
from IPython import get_ipython
import matplotlib.pyplot as plt
from globals import path_results, path_figures, path_info_sub

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------

warnings.filterwarnings("ignore")
get_ipython().magic('reset -sf')

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
#finalDF['sex'] = df['sex'].map({'F': 0, 'M': 1})
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
            "BMI",]

# Count the number of NaN values per subject (row)
nan_counts = beh_data.isna().sum(axis = 1)
# Create unified cleaned dataset (males + females with ≤ 20 NaNs)
mask_nan = finalDF.isna().sum(axis=1) <= 20
finalDF['sex'] = sex  # Add sex info
finalDF_filtered = finalDF[mask_nan].copy()
finalDF_filtered.fillna(finalDF_filtered.mode().iloc[0], inplace=True)

# Convert age from months to years
finalDF_filtered['age_years'] = finalDF_filtered['interview_age'] / 12

# Separate males and females
females = finalDF_filtered[finalDF_filtered['sex'] == 0]
males = finalDF_filtered[finalDF_filtered['sex'] == 1]

# Behavioral names
behavioral_names = name_write[1:]  # skip 'age'

# Plot
cmap = plt.get_cmap('coolwarm')
colors = cmap(np.linspace(0, 1, 100))
plt.figure(figsize=(22, 28))
for i, name in enumerate(behavioral_names):
    plt.subplot(7, 4, i + 1)
    plt.scatter(females['age_years'], females.iloc[:, i + 1], color=colors[90], 
                label='Female',
                alpha=0.7, s=40)
    plt.scatter(males['age_years'], males.iloc[:, i + 1], color=colors[10], 
                label='Male',
                alpha=0.7, s=40)
    plt.ylabel(name)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.title(f"{name} vs. Age")
    if i == 0:
        plt.legend()
plt.tight_layout()
plt.savefig(path_figures + "physiological_vs_age_scatterplots_female_male.svg", dpi=300)
plt.show()

#------------------------------------------------------------------------------
# Compute relevance to age - for male group
#------------------------------------------------------------------------------

from scipy.stats import spearmanr

n_perm = 1000  # Number of permutations

plt.figure(figsize=(22, 28))
for i, name in enumerate(behavioral_names):
    plt.subplot(7, 4, i + 1)
    
    x = males['age_years'].values
    y = males.iloc[:, i + 1].values
    
    # Remove any rows with NaNs
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]

    plt.scatter(x, y, color=colors[10], label='Male', alpha=0.7, s=40)
    r_obs, _ = spearmanr(x, y)

    # Permutation test: shuffle y and compute r each time
    r_perm = np.zeros(n_perm)
    for j in range(n_perm):
        y_perm = np.random.permutation(y)
        r_perm[j], _ = spearmanr(x, y_perm)

    p_perm = np.mean(np.abs(r_perm) >= np.abs(r_obs))

    plt.text(0.05, 0.95, f"ρ = {r_obs:.2f}\np = {p_perm:.3g}",
             transform=plt.gca().transAxes,
             verticalalignment='top', fontsize=10,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    plt.ylabel(name)
    plt.title(f"{name} vs. Age")
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    if i == 0:
        plt.legend()

plt.tight_layout()
plt.show()

#------------------------------------------------------------------------------
# END
