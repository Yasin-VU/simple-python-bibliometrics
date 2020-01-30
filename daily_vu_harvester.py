## Daily VU harvester
#
# Harvest ScopusSearch for today
# enriches and cleans it
# then pushes it to the database with incremented primary key
#
# warning: the code uses 'today' so you have to run it before midnight
#
# warning: code of harvester should be refactored and bound with oadash

## imports
import pandas as pd
from pybliometrics.scopus import ScopusSearch
# imports from our own import framework
######import sys
#####sys.path.insert(0, 'C:/Users/yasin/Desktop/git/common_functions')  # not needed sometimes
from nlp_functions import faculty_finder
from nlp_functions import corresponding_author_functions
#
from core_functions import add_year_and_month
from core_functions import get_scopus_abstract_info
from core_functions import get_first_chosen_affiliation_author  # ! i want all vu authors now : )
from core_functions import add_unpaywall_columns
from core_functions import add_deal_info
from core_functions import fn_cats, get_today, get_today_for_pubdatetxt
from database_functions import get_last_primary_key, push_df_to_db, get_connection, \
    process_df_to_list_to_push, pre_process_for_push

## settings
running_on_server = False
#
# paths
if running_on_server:
    df_pass = pd.read_csv(r'C:\Users\yasing\Desktop\git\password_mydb.csv')
    path_deals = 'C:/Users/yasing/Desktop/oa oktober/apcdeals.csv'                #check
    path_isn = 'C:/Users/yasing/Desktop/oa oktober/ISN_ISSN.csv'                  #check
    path_org = 'C:/Users/yasing/Desktop/oa oktober/vu_organogram_2.xlsx'          #check
    path_out = 'C:/Users/yasing/Desktop/oa oktober/'                              #check
    path_vsnu_afids = 'C:/Users/yasing/Desktop/oa oktober/afids_vsnu_nonfin.csv'  #check
else:
    df_pass = pd.read_csv(r'C:\Users\yasin\Desktop\git\password_mydb.csv')
    path_deals = r'G:\UBVU\Data_RI\raw data algemeen\apcdeals.csv'
    path_isn = r'G:\UBVU\Data_RI\raw data algemeen\ISN_ISSN.csv'
    path_org = r'G:\UBVU\Data_RI\raw data algemeen\vu_organogram_2.xlsx'
    path_out = 'C:/Users/yasin/Desktop/oa new csv/'  # no r
    path_vsnu_afids = r'G:\UBVU\Data_RI\raw data algemeen\afids_vsnu_nonfin.csv'
#
# read in pass (distribute mysql accounts as some point)
user = df_pass.user[0]
pw = df_pass.pw[0]
host = df_pass.host[0]
database = df_pass.database[0]
#
chosen_affid = ["60008734","60029124","60012443","60109852","60026698","60013779","60032886","60000614",
                "60030550","60013243","60026220","60001997"]  # I added 60001997 and thus I added VUMC
# corresponding author
vu_afids = chosen_affid
#this is vsnu w/o phtu and such (borrowed from VSNU-SDG-data), but should approach the UKB list... good for now. update later.
all_vsnu_sdg_afids = pd.read_csv(path_vsnu_afids).iloc[:,1].astype('str').to_list()
#


# harvester
# warning: the code uses 'today' so you have to run it before midnight
#
# harvest from ScopusSearch everything from VU+VUMC of today
# because the API has issues with direct today command we instead take entire month and isolate today
#
# prepare query
VU_with_VUMC_affid = "(   AF-ID(60001997) OR    AF-ID(60008734) OR AF-ID(60029124) OR AF-ID(60012443) OR AF-ID(60109852) OR AF-ID(60026698) OR AF-ID(60013779) OR AF-ID(60032886) OR AF-ID(60000614) OR AF-ID(60030550) OR AF-ID(60013243) OR AF-ID(60026220))"
my_query = VU_with_VUMC_affid + ' AND ' + "PUBDATETXT( " + get_today_for_pubdatetxt() + " )"  # RECENT(1) is somehow very slow
print(my_query)
#
# call the scopussearch API
s = ScopusSearch(my_query, refresh=True, download=True)
df = pd.DataFrame(s.results)
#
# filter to records of today
today = get_today()
df = df[df.coverDate == today]
#
# here is the result (may be empty on some days)
###df

# pre-processing aspect
# we need to add extra sources, clean it, rename columns and make it ready for push
# this is a static copy, and you should migrate processing to pycharm and call that from here


# Add info on year and month
df = add_year_and_month(df, 'coverDate')  # add info columns

# prepare the faculty_finder NLP tool
org_info = pd.read_excel(path_org, skiprows=0)
ff = faculty_finder(organizational_chart=org_info)

# Per EID, get scopus abstract info, get first vu author and use NLP to find faculty
# initialize
df_ab = pd.DataFrame()
df_au = pd.DataFrame()
df_ff = pd.DataFrame()
for counter, cur_eid in enumerate(df.eid.tolist()):

    print('getting abstract info for ' + str(counter + 1) + ' out of ' + str(len(df.eid.tolist())))

    # get abstract
    dict_ab_info = get_scopus_abstract_info(cur_eid)  # !
    dict_ab_info['eid'] = cur_eid

    # get first chosen affiliation author
    dict_auth_info = get_first_chosen_affiliation_author(dict_ab_info['abstract_object'], chosen_affid)
    dict_auth_info['eid'] = cur_eid

    # get faculty
    if dict_auth_info['first_affil_author_has_error'] == True:
        print('no chosen affid author found at EID:' + str(cur_eid))
        dict_ff = ff.match_nan()
    else:
        # get faculty
        dict_ff = ff.match(dict_auth_info['first_affil_author_org'])
    dict_ff['eid'] = cur_eid

    df_ab = df_ab.append(dict_ab_info, ignore_index=True)
    df_au = df_au.append(dict_auth_info, ignore_index=True)
    df_ff = df_ff.append(dict_ff, ignore_index=True)

df = df.merge(df_ab, on='eid', how='left')
df = df.merge(df_au, on='eid', how='left')
df = df.merge(df_ff, on='eid', how='left')

# add unpaywall info
df = add_unpaywall_columns(df, silent=False)  # !

# add deal info
df = add_deal_info(path_deals=path_deals, path_isn=path_isn, df_b=df)

# add corresponding author info
df = (corresponding_author_functions()
      .add_corresponding_author_info(df=df,
                                     vu_afids=vu_afids,
                                     ukb_afids=all_vsnu_sdg_afids))

# futher process
df['upw_oa_color_category'] = df.upw_oa_color.apply(fn_cats)

## get connection to db

connection = get_connection(host=host,
                            database=database,
                            user=user,
                            pw=pw)

## now pre-process specifically for push to database

# we make primary keys here, so let's find the last one
last_primary_key, success = get_last_primary_key(connection)
last_primary_key, success

df = pre_process_for_push(df, last_primary_key)

list_to_push = process_df_to_list_to_push(df)

# I don't like closing connection after every command, edit that out please, make external closer


# actually push it

connection = get_connection(host=host,
                            database=database,
                            user=user,
                            pw=pw)
push_df_to_db(connection, df)

print('done')


