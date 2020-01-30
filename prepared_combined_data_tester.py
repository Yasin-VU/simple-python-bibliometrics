#
# goal: test prepare_combined_data

# imports
import pandas as pd
import numpy as np
import sys
from core_functions import prepare_combined_data


prepare_combined_data(start_path=r'G:\UBVU\Data_RI\raw data algemeen\pipeline_test',
                         year_range=(2017, 2018, 2019),
                         add_abstract=True,
                         skip_preprocessing_pure_instead_load_cache=False,  # safe
                         remove_ultra_rare_class_other=True,
                         path_pw=r'C:\Users\yasin\Desktop\git',
                         org_info=pd.read_excel(r'G:\UBVU\Data_RI\raw data algemeen\vu_organogram_2.xlsx', skiprows=0))


# I am doing to rest-run right now
# afterwards commit what you have and continue

# it works!



