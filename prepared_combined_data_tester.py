#
# goal: test prepare_combined_data

# imports
from static import PATH_START, PATH_START_PERSONAL
from static import PATH_START_SERVER , PATH_START_PERSONAL_SERVER
import pandas as pd
import numpy as np
import sys
from core_functions import prepare_combined_data


print('beep')
pd.read_csv(r'G:\UBVU\Data_RI\raw data algemeen\oa2019map\scopus_processed\EXAMPLE_knip_OA_VU2018_met_corresponding_authors.csv')
print('beep')

prepare_combined_data(start_path=PATH_START + '/raw data algemeen/oa2019map',
                      year_range=(2018, 2019, 2020),
                      add_abstract=True,
                      skip_preprocessing_pure_instead_load_cache=False,  # safe
                      remove_ultra_rare_class_other=True,
                      path_pw=PATH_START_PERSONAL,
                      org_info=pd.read_excel(PATH_START + '/raw data algemeen/vu_organogram_2.xlsx', skiprows=0))

print('done')
# worked once
# now trying to make it work again for the current situation
# some filenames have changed

