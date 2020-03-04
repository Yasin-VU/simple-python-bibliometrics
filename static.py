# static.py
#
# The goal of this static file is to gather all settings such as paths to one central point,
# and to have the test/production-switch right here and nowhere else
# all variables in static.py must be full-uppercase

import pandas as pd

USE_MULTI_THREAD_DEFAULT = True  # does nothing for now

PATH_START =  'G:/UBVU/Data_RI/' #'/media/sf_Data_RI/'  #
PATH_START_PERSONAL = 'C:/Users/yasin/Desktop/git'
#
PATH_START_SERVER = 'C:/Users/yasing/Desktop/oa oktober/'
PATH_START_PERSONAL_SERVER = 'C:/Users/yasing/Desktop/git'


PRODUCTION_MODE = False  # does nothing yet

SCOPUS_KEYS = pd.read_csv(PATH_START_PERSONAL + '/scopuskeys.txt').iloc[:, 0].to_list()

# the email to provide to unpaywall.org, please use your own : )
UNPAYWALL_EMAIL = 'b.gunes@vu.nl'

# path to unpaywall invalid-reponse-object cache  # this will be made easier to use in an upcoming release
PATH_STATIC_RESPONSES = PATH_START + 'raw data algemeen/api_caches/upw_invalid_request_object.pkl'
PATH_STATIC_RESPONSES_ALTMETRIC = PATH_START + 'raw data algemeen/api_caches/upw_invalid_request_object_alt.pkl'
PATH_STATIC_RESPONSES_SCOPUS_ABS = PATH_START + 'raw data algemeen/api_caches/upw_invalid_request_object_scopus_abs.pkl'

# DEFAULT maximum number of workers for multi-threading
MAX_NUM_WORKERS = 8  # if you experience any issues, please first reduce this to 16 or 8 and retry

