"""

Show demographics per dataset:

    ----------------------------------
    total number of ASL images
    1305
    total number of females
    718
    total number of males
    587
    ----------------------------------
    total number of ASL images - HCP-D
    627
    total number of females
    337
    total number of males
    290
    age range is:
    5.583333333333333
    21.916666666666668
    ----------------------------------
    total number of ASL images - HCP-A
    678
    total number of females
    381
    total number of males
    297
    age range is:
    36.0
    100.0
    ----------------------------------

"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import warnings
import numpy as np
import pandas as pd
from IPython import get_ipython
from globals import path_info_sub

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
ageold = np.array(df['interview_age'])/12
sexold = df.sex

df = pd.read_csv(path_info_sub + 'Dev_clean_data_info.csv')
num_subjects = len(df)
agedev = np.array(df['interview_age'])/12
sexdev = df.sex

sex = np.concatenate((sexdev, sexold))
age = np.concatenate((agedev, ageold))

print('----------------------------------')
print('total number of ASL images')
print(len(age))
print('total number of females')
print(sum(sex == 'F'))
print('total number of males')
print(sum(sex == 'M'))
print('----------------------------------')
print('total number of ASL images - HCP-D')
print(len(agedev))
print('total number of females')
print(sum(sexdev == 'F'))
print('total number of males')
print(sum(sexdev == 'M'))
print('age range is:')
print(min(agedev))
print(max(agedev))
print('----------------------------------')
print('total number of ASL images - HCP-A')
print(len(sexold))
print('total number of females')
print(sum(sexold == 'F'))
print('total number of males')
print(sum(sexold == 'M'))
print('age range is:')
print(min(ageold))
print(max(ageold))
print('----------------------------------')

#------------------------------------------------------------------------------
# END