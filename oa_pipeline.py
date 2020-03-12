

# I want to have the oa2019 stuff in 1 central place, but w/o luigi it is HORRIBLY inefficient
# you will redo stuff a lot
# and it is very slow as it has no multi-node nor multi-thread
#
# let's refactor this...

import pandas as pd
from oa2019 import get_scopus_arm
from core_functions import prepare_combined_data
from static import PATH_START, PATH_START_PERSONAL
from static import PATH_START_SERVER , PATH_START_PERSONAL_SERVER




# this is a mess: you need 3 years for prepare_combined_data, but you do not want to refresh it
# or you may want to dl more years first...
#
# ISSUE: the code below needs to be variabilized to year_ranges and settings at top should be propogated
#        right now it will only work for A SINGLE COMBINATION OF SETTINGS
#        and that is not satisfactory
#        also, test it



# Step 1
#
# be careful with this function because it ignores and overwrites existing years: it is not luigi (!)
MY_YEARSET = [2010, 2011, 2012, 2013]
get_scopus_arm(MY_YEARSET=MY_YEARSET)

# Step 2
#
MY_YEARSET2 = [2010, 2011, 2012]
prepare_combined_data(start_path=PATH_START + '/raw data algemeen/oa2019map',
                      year_range=tuple(MY_YEARSET2),  # please chunks of 3 year or it will not work!
                      add_abstract=True,
                      skip_preprocessing_pure_instead_load_cache=False,  # safe
                      remove_ultra_rare_class_other=True,
                      path_pw=PATH_START_PERSONAL,
                      org_info=pd.read_excel(PATH_START + '/raw data algemeen/vu_organogram_2.xlsx', skiprows=0))

# Step 3
#
from nlp_functions import SoftTitleMatcher
# and run it, make it a function and refactor it
core = 'oa2019map'
df_total = pd.read_csv(PATH_START + '/raw data algemeen/' + core + '/merged_data/df_total.csv')
chosen_year = 2019
stm = SoftTitleMatcher()
df_total_with_STM, df_total_with_STM_rich_2018 = stm.improve_merged_table_using_STM_results(df_total=df_total,
                                                                                            chosen_year=chosen_year,
                                                                                            out_path=None,
                                                                                            do_save=False,
                                                                                            cond_len=4,
                                                                                            cond_score=0.6)
df_total_with_STM_rich_2018.to_csv(PATH_START + '/raw data algemeen/' + core + '/merged_data/refactor_test.csv')


# Step 4
#
import data_integration_column_fuser
# and run it, make it a function and refactor it
from data_integration_column_fuser import column_fuser_and_fac_unknown_fixer
column_fuser_and_fac_unknown_fixer()

# Step 5: explicitly return the resulting dataframe or at least print its path man
print('all done')











