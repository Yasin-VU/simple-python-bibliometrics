#
# This is a single file containing all routines to run oa for any year
# It needs some steps:
# 1. all year settings need to be pulled out of the subroutines
# 1b. AND THE PATH 
# 2. all subroutines must be turned into pure function
# 3. the settings should be logically pulled to the top and simplified
#    it is OK to reduce the scope to work for exactly 1 year
#    we can add a loop later for multiple oa-years, or let luigi deal with it
# 4. move everything to luigi
# 5. refactor again within luigi
# 6. refactor to allow multi oa-years (add another arm in luigi for that)

# what are pure functions?
# pure functions are state-less functions
# for the same input they always generate the same output
# this also means I have to make them of a form where inputs are fed in
# s.t. no files are read (since that would not be pure)
# for now you can ignore organogram and pw for this aspect
#
# also we need to feed in start_path to isolate projects
# and for storing intermediate outputs and final outputs
# 
# and we need to feed in the most critical settings: the year(-ranges)
#
# also, once this works, the i/o files should be named appropriately with yrs!
#
# these should be the first to happen
#
#
# just take your time and refactor everything well s.t. it will work well
# not just now but also in the future and for arbitrary year choices
# also, it will cut down future development costs because it will be clean
# last time we did not refactor well and had to spend 3 days on it
# to get a fresh year and that timedelay is problematic
#
#
# there are like 10 refactor tasks or so
# so just do it part by part
# do one thing at a time 
#




# imports
import pandas as pd
from static import PATH_START, PATH_START_PERSONAL
from static import PATH_START_SERVER , PATH_START_PERSONAL_SERVER
#
from oa2019 import get_scopus_arm
from core_functions import prepare_combined_data
from nlp_functions import SoftTitleMatcher
from data_integration_column_fuser import column_fuser_and_fac_unknown_fixer

# top-level settings (requires refactoring first)
top_start_path = PATH_START + '/raw data algemeen/oa2019map'
top_chosen_year = 2019





# Step 1
dict_output = get_scopus_arm(MY_YEARSET=[top_chosen_year-1, top_chosen_year, top_chosen_year+1],
                             start_path_with_slash = top_start_path + '/',
                             df_in=None, # there is no df_in (!))
                             do_save=False)  
# dict_output has three keys, one for every year, and contains oa_knip dataframes

# warning: using do_save above will break Step 2 unless you refactor it

# Step 2
prepare_combined_data(start_path=top_start_path,
                      year_range=tuple([top_chosen_year-1, top_chosen_year, top_chosen_year+1]),  # please chunks of 3 year or it will not work!
                      add_abstract=True,
                      skip_preprocessing_pure_instead_load_cache=False,  # safe
                      remove_ultra_rare_class_other=True,
                      path_pw=PATH_START_PERSONAL,
                      org_info=pd.read_excel(PATH_START + '/raw data algemeen/vu_organogram_2.xlsx', skiprows=0))

# Step 3
stm = SoftTitleMatcher()
df_total = pd.read_csv(top_start_path + '/merged_data/df_total.csv')  # this must be generated in step 2 above... this needs a year-range in naming csv
df_total_with_STM, df_total_with_STM_rich_chosen_year = stm.improve_merged_table_using_STM_results(df_total=df_total,
                                                                                            chosen_year=top_chosen_year,
                                                                                            out_path=None,
                                                                                            do_save=False,
                                                                                            cond_len=4,
                                                                                            cond_score=0.6)

# Step 4
df_final = column_fuser_and_fac_unknown_fixer(df = df_total_with_STM_rich_chosen_year,
                                              chosen_year=top_chosen_year,
                                              start_path = top_start_path,
                                              do_save = False)
    
# Step 5: explicitly return the resulting dataframe or at least print its path man
df_final.to_csv(top_start_path + '/open_access_dashboard_data_v3_' + str(top_chosen_year) + '.csv' )
    
print('all done')











