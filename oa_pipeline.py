

# I want to have the oa2019 stuff in 1 central place, but w/o luigi it is HORRIBLY inefficient
# you will redo stuff a lot
# and it is very slow as it has no multi-node nor multi-thread
#
# let's refactor this...


from oa2019 import get_scopus_arm
from core_functions import prepare_combined_data
from static import PATH_START, PATH_START_PERSONAL
from static import PATH_START_SERVER , PATH_START_PERSONAL_SERVER



MY_YEARSET = [2010, 2011, 2012, 2013]
# this is a mess: you need 3 years for prepare_combined_data, but you do not want to refresh it
# or you may want to dl more years first...

MY_YEARSET2 = [2010, 2011, 2012]




# be careful with this function because it ignores and overwrites existing years: it is not luigi (!)
get_scopus_arm(MY_YEARSET=MY_YEARSET)


prepare_combined_data(start_path=PATH_START + '/raw data algemeen/oa2019map',
                      year_range=tuple(MY_YEARSET2),  # please chunks of 3 year or it will not work!
                      add_abstract=True,
                      skip_preprocessing_pure_instead_load_cache=False,  # safe
                      remove_ultra_rare_class_other=True,
                      path_pw=PATH_START_PERSONAL,
                      org_info=pd.read_excel(PATH_START + '/raw data algemeen/vu_organogram_2.xlsx', skiprows=0))


import tester_soft_title_matcher
# and run it, make it a function and refactor it

import data_integration_column_fuser
# and run it, make it a function and refactor it












