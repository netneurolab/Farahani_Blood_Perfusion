"""

Clean the physiological measures in the HCP-A dataset
Preprocess the raw data for PLS analysis.

"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import warnings
import numpy as np
import pandas as pd
from IPython import get_ipython
from globals import path_measures, path_info_sub, path_results

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------

warnings.filterwarnings("ignore")
get_ipython().magic('reset -sf')

#------------------------------------------------------------------------------
# Load blood test result dataset
#------------------------------------------------------------------------------

name_measure = 'bsc01.txt'
df_bsc01 = pd.read_csv(path_measures + name_measure, delimiter = '\t')
df_bsc01 = df_bsc01.drop(columns = ['collection_id',
                                  'bsc01_id',
                                  'dataset_id',
                                  'interview_date',
                                  'comqother',
                                  'collection_title',
                                  'interview_age',
                                  'sex'])

combined_df = df_bsc01

#------------------------------------------------------------------------------
# Load and clean blood pressure dataset (vitals01)
#------------------------------------------------------------------------------

name_measure = 'vitals01.txt'
df_vitals01 = pd.read_csv(path_measures + name_measure, delimiter = '\t')
df_vitals01 = df_vitals01.drop(columns = ['collection_id',
                                  'vitals01_id',
                                  'dataset_id',
                                  'interview_date',
                                  'collection_title',
                                  'interview_age',
                                  'sex'])
df_vitals01[['bp_stand_1', 'bp_stand_2']] = df_vitals01['bp_stand'].str.split('/', expand = True)
df_vitals01[['bp_1', 'bp_2']] = df_vitals01['bp'].str.split('/', expand = True)

# Convert these columns to numeric
df_vitals01['bp_1'] = pd.to_numeric(df_vitals01['bp_1'], errors = 'coerce')
df_vitals01['bp_2'] = pd.to_numeric(df_vitals01['bp_2'], errors = 'coerce')
df_vitals01['MAP'] = df_vitals01['bp_2'] + ((1/3) * (df_vitals01['bp_1'] - df_vitals01['bp_2']))
df_vitals01['PP'] =  (df_vitals01['bp_1'] - df_vitals01['bp_2'])

combined_df = pd.merge(combined_df, df_vitals01, on = 'subjectkey', how = 'outer')
print(len(combined_df))

#------------------------------------------------------------------------------
# Add more information to the column names
#------------------------------------------------------------------------------

# Extract the first row values
first_row_values = combined_df.iloc[0]
new_column_names = [f"{col} ({val})" for col, val in zip(combined_df.columns, first_row_values)]
combined_df.columns = new_column_names
combined_df = combined_df.drop(0).reset_index(drop=True)
print(combined_df)
combined_df = combined_df.rename(columns = {'subjectkey (The NDAR Global Unique Identifier (GUID) for research subject)': 'subjectkey'})

#------------------------------------------------------------------------------
# Merge with basic subject information
#------------------------------------------------------------------------------

df = pd.read_csv(path_info_sub + 'clean_data_info.csv')
columns_to_keep = ['subjectkey', 'sex', 'interview_age']
df = df[columns_to_keep]
a = pd.merge(df, combined_df, on = 'subjectkey', how = 'left')

#------------------------------------------------------------------------------
# Clean the insulin column
#------------------------------------------------------------------------------

data = a['insomm (Insulin: Comments)']

# Remove the ' uU/mL' part from each string and convert to float
cleaned = data.str.replace(' uU/mL', '', regex=False)

# Now 'cleaned' should contain only numeric strings (or NaN). Convert to float:
a['insomm (Insulin: Comments)'] = pd.to_numeric(cleaned, errors = 'coerce')

#------------------------------------------------------------------------------
# Function to clean DataFrame by removing columns that cannot be converted to numeric (excluding the first row)

def clean_dataframe(df):
    columns_to_drop = []
    for col in df.columns:
        if col == 'subjectkey':
            continue
        series_to_check = df[col].iloc[0:]
        try:
            pd.to_numeric(series_to_check, errors = 'raise')
        except ValueError:
            columns_to_drop.append(col)
        except TypeError:
            columns_to_drop.append(col)

        columns_to_drop.append('bld_core_d2ph (All Tubes Time from Draw to Processing: Hour(s):)')
        columns_to_drop.append('bld_core_d2pm (All Tubes Time from Draw to Processing: Minute(s):)')
        columns_to_drop.append('bld_core_p2fh (All Tubes Time from Processing to Freezing: Hour(s):)')
        columns_to_drop.append('bld_core_p2fm (All Tubes Time from Processing to Freezing: Minute(s):)')
        columns_to_drop.append('bld_core_grn (CORE: Number of Tubes Collected: Green Tubes)')
        columns_to_drop.append('bld_core_snack (Fasting: Time of last meal/snack)')
        columns_to_drop.append('bld_core_spin (Spin at 1000 - 1300G for 10 mins)')
        columns_to_drop.append('biospc_8 (Number of Purple tubes collected)')
        columns_to_drop.append('biospc_6 (Number of Yellow tubes collected)')
        columns_to_drop.append('biospc2_purple_tubes (num of purple tubes collected sample 2)')
        columns_to_drop.append('fasting (fasting blood draw y/n)')
        columns_to_drop.append('ed1_saliva (SALIVA SAMPLE COLLECTED)')

    # Drop columns that failed conversion
    df_cleaned = df.drop(columns=columns_to_drop)
    return df_cleaned

cleaned_df = clean_dataframe(a)
print(cleaned_df)

def clean_dataframe(df):
    columns_to_drop = []
    for col in df.columns:
        if col == 'subjectkey':
            continue
        # Calculate the percentage of NaN values excluding the first row
        nan_percentage = df[col].iloc[0:].isna().mean() * 100
        # Check if more than 50% of the column is NaN
        if nan_percentage > 50:
            columns_to_drop.append(col)
            continue
        # Calculate the number of unique values in the column
        unique_values = np.unique(df[col].dropna())
        if len(unique_values) < 2:
            columns_to_drop.append(col)
            continue
    # Drop columns that failed conversion or have too many NaNs
    df_cleaned = df.drop(columns = columns_to_drop)
    return df_cleaned
finalDF = clean_dataframe(cleaned_df)

# Save the final cleaned dataset for PLS analysis
finalDF.to_csv(path_results + 'finalDF.csv', index = False)

#------------------------------------------------------------------------------
# END
