# static.py
#
# The goal of this static file is to gather all settings such as paths to one central point,
# and to have the test/production-switch right here and nowhere else
# all variables in static.py must be full-uppercase



PATH_START = 'G:/UBVU/Data_RI/'
PATH_START_PERSONAL = r'C:\Users\yasin\Desktop\git'
PATH_START_PERSONAL_SERVER = r'C:\Users\yasing\Desktop\git'
PATH_START_SERVER = 'C:/Users/yasing/Desktop/oa oktober'


PRODUCTION_MODE = False  # does nothing yet

# the email to provide to unpaywall.org
UNPAYWALL_EMAIL = 'b.gunes@vu.nl'

# path to unpaywall invalid-reponse-object cache
PATH_STATIC_RESPONSES = PATH_START + 'raw data algemeen/api_caches/upw_invalid_request_object.pkl'
PATH_STATIC_RESPONSES_ALTMETRIC = PATH_START + 'raw data algemeen/api_caches/upw_invalid_request_object_alt.pkl'
PATH_STATIC_RESPONSES_SCOPUS_ABS = PATH_START + 'raw data algemeen/api_caches/upw_invalid_request_object_scopus_abs.pkl'

# DEFAULT maximum number of workers for multi-threading
MAX_NUM_WORKERS = 32  # if you experience any issues, please first reduce this to 16 or 8 and retry

