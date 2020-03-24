
# Goal: run the baseline code for open access dashboard 2019 data, including recent patches
#
# do not multi-thread this as we want to be MINIMALLY-INVASIVE
# why? because we want to use the verified and tested baseline code
# the recent patches are only included to cope with API serverside changes like Scival delimiters

# imports
import pandas as pd
import numpy as np
from pybliometrics.scopus import ScopusSearch
import nltk
###nltk.download('punkt')

# imports from our own import framework 
import sys
###sys.path.insert(0, 'C:/Users/yasing/Desktop/git/simple-python-bibliometrics')  # not needed in pycharm
from nlp_functions import faculty_finder
from nlp_functions import get_dist
from nlp_functions import corresponding_author_functions
#
from core_functions import add_year_and_month
from core_functions import get_scopus_abstract_info
from core_functions import get_first_chosen_affiliation_author  # ! i want all vu authors now : )
from core_functions import add_unpaywall_columns
from core_functions import my_timestamp
from core_functions import add_deal_info
### this needs to be replaced with the luigi-scopus-arm as that works 100x better, but first this...
# !


def get_scopus_arm(MY_YEARSET,
                   start_path_with_slash,
                   df_in=None, # there is no df_in (!))
                   do_save=False): 
    """
    Use the assumption that MY_YEARSET is always 3 years
    Once we get this in Luigi it will work better than arbitrary length sets
    because this is an ATOMIC split of work and works well concurrently
    luigi will always skip parts if they already exist
    you do have to put it well in luigi: this function will be 2 pipe-types
    type-1 will do 1 year only
    type-2 will combine 3 years only
    and that is all you need because the entire pure arm is for 1 chosen year
    but can be easily extended to do multiple chosen years efficiently
    """
        
    dict_output = {}

    for MY_YEAR in MY_YEARSET:

        print(MY_YEAR)
        # settings

        # testing
        override_query_for_testing = False
        running_on_server = False

        # paths
        if running_on_server:
            path_deals = 'C:/Users/yasing/Desktop/oa oktober/apcdeals.csv'                #check
            path_isn = 'C:/Users/yasing/Desktop/oa oktober/ISN_ISSN.csv'                  #check
            path_org = 'C:/Users/yasing/Desktop/oa oktober/vu_organogram_2.xlsx'          #check
            path_out = start_path_with_slash #'C:/Users/yasing/Desktop/oa oktober/'                              #check
            path_vsnu_afids = 'C:/Users/yasing/Desktop/oa oktober/afids_vsnu_nonfin.csv'  #check
        else:
            path_deals = r'G:\UBVU\Data_RI\raw data algemeen\apcdeals.csv'
            path_isn = r'G:\UBVU\Data_RI\raw data algemeen\ISN_ISSN.csv'
            path_org = r'G:\UBVU\Data_RI\raw data algemeen\vu_organogram_2.xlsx'
            path_out = start_path_with_slash #'C:/Users/yasin/Desktop/oa new csv/'  # no r
            path_vsnu_afids = r'G:\UBVU\Data_RI\raw data algemeen\afids_vsnu_nonfin.csv'

        # scopus search and affiliation
        #
        # ! VUMC HAS BEEN ADDED !
        #
        chosen_affid = ["60008734","60029124","60012443","60109852","60026698","60013779","60032886","60000614",
                        "60030550","60013243","60026220","60001997"]  # I added 60001997 and thus I added VUMC
        #VU_noMC_affid = "(AF-ID(60008734) OR AF-ID(60029124) OR AF-ID(60012443) OR AF-ID(60109852) OR AF-ID(60026698) OR AF-ID(60013779) OR AF-ID(60032886) OR AF-ID(60000614) OR AF-ID(60030550) OR AF-ID(60013243) OR AF-ID(60026220))"
        VU_with_VUMC_affid = "(   AF-ID(60001997) OR    AF-ID(60008734) OR AF-ID(60029124) OR AF-ID(60012443) OR AF-ID(60109852) OR AF-ID(60026698) OR AF-ID(60013779) OR AF-ID(60032886) OR AF-ID(60000614) OR AF-ID(60030550) OR AF-ID(60013243) OR AF-ID(60026220))"
        my_query = VU_with_VUMC_affid + ' AND  ' + "( PUBYEAR  =  " + str(MY_YEAR) +" )"   ### "PUBDATETXT(February 2018)"

        # TITLE(TENSOR) AND

        # corresponding author
        vu_afids = chosen_affid
        # this is vsnu w/o phtu and such (borrowed from VSNU-SDG-data), but should approach the UKB list... good for now. update later.
        all_vsnu_sdg_afids = pd.read_csv(path_vsnu_afids).iloc[:, 1].astype('str').to_list()

        # testing
        if override_query_for_testing:
            my_query = 'TITLE(TENSOR LPV)'
            print('overriding query for testing')


        # ETLMIG MIGRATION DONE


        # helper functions
        # ! CAREFUL! COPIED CODE
        def fn_cats(row):
            if row == 'closed':
                result = 1
            elif row == 'hybrid':
                result = 2
            elif row == 'bronze':
                result = 3
            elif row == 'green':
                result = 4
            elif row == 'gold':
                result = 5
            else:
                result = 0  # nans etc
            return result

        # entire pipeline

        # Perform ScopusSearch
        s = ScopusSearch(my_query, refresh=True)  #(VU_aff + " AND " + recent, refresh=True)
        df = pd.DataFrame(s.results)

        # Remove unnecessary columns
        fav_fields = ['eid',  'creator',  'doi',  'title',  'afid',
         'affilname',  'author_count',  'author_names',  'author_afids',
         'coverDate',  'coverDisplayDate',  'publicationName', 'issn',  'source_id', 'eIssn',
         'citedby_count', 'fund_sponsor', 'aggregationType', 'openaccess']
        df = df[fav_fields]  # cut fields

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

            print('getting abstract info for ' + str(counter+1) + ' out of ' + str(len(df.eid.tolist())))

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

        df = df.merge(df_ab,on='eid',how='left')
        df = df.merge(df_au,on='eid',how='left')
        df = df.merge(df_ff,on='eid',how='left')

        print('df_ab,au,ff done')
        #df.to_csv(r'C:\Users\yasing\Desktop\oa oktober\oa' + my_timestamp() + '.csv')
        # df.to_pickle(path_out + 'oa_base_' + my_timestamp() + str(MY_YEAR) + '.pkl')

        # add unpaywall info
        df = add_unpaywall_columns(df, silent=False)   # !

        # add deal info
        df = add_deal_info(path_deals=path_deals, path_isn=path_isn, df_b=df)

        # add corresponding author info
        df = (corresponding_author_functions()
              .add_corresponding_author_info(df=df,
                                             vu_afids=vu_afids,
                                             ukb_afids=all_vsnu_sdg_afids))

        # post-process
        df['upw_oa_color_category'] = df.upw_oa_color.apply(fn_cats)
        df['upw_oa_color_verbose'] = df['upw_oa_color'].apply(lambda x: 'unknown' if x is np.nan else x)

        # save it
        # save to pickle with abstract_object, for now
        # df.to_pickle(path_out  + 'oa' + my_timestamp() + str(MY_YEAR) +  '.pkl')
        # save to csv without abstract_object0
        if do_save:
            df.drop(columns=['abstract_object']).to_csv(path_out + 'oa' + my_timestamp() + str(MY_YEAR) +  '.csv')


        # diagnose
        # verval-analyse
        print('verval-analyse')
        print('aantal scopus publicaties: ' + str(len(df)))
        print('api error: abstract API: ' + str( len(df[df.abstract_error_message == 'abstract api error']) ))
        print('api error: authgroup/afdelinginfo: ' + str( df.no_author_group_warning.sum() ))  # ab.authgroup error
        print('api error: authgroup.x/afdelinginfo details: ' + str( len(df[df.first_affil_author_has_error == True]) )) # ab.authgroup ok, error deeper in it
        print('api missing data: data afdelingsinfo ontbreekt no1: ' + str( len(df[(df.first_affil_author == None) & (df.first_affil_author_has_error == False)]) ))
        print('api missing data: data afdelingsinfo ontbreekt no2: ' + str( len(df[df.first_affil_author_org == None]) ))
        # pas hier heb je data om mee te werken
        print('no match: no faculty name match and bag of words only has trivial words (zoals lidwoorden en Amsterdam): ' + str( len(df[df.ff_message == 'no faculty name match and bag of words only has trivial words']) ))
        print('no match: no faculty name match and no bag of words match despite non-trivial words (vaak VUMC, soms typo): ' + str( len(df[df.ff_message == 'no faculty name match and no bag of words match despite non-trivial words']) ))
        print('aantal matches: ' + str( len(df[df.ff_score>0]) ))
        # diagnostics can be improved further by capturing the last 6 fails too

        # print done
        print('done')



        # extra: post-process

        ##df = pd.read_csv(r'C:\Users\yasin\Desktop\oa new csv\OA_VU2018_met_corresponding_authors.csv')
        ##list(df)

        # this also drop abstract_object(!)
        df2 = df[[ 'eid',
         'doi',
         'title',
         'year',
         'publicationName',
         'issn',
         'eIssn',
         'fund_sponsor',
         'aggregationType',
         'first_affil_author',
         'first_affil_author_org',
         'ff_match',
         'ff_match_subgroup',
         'ff_message',
         'ff_provided_organization_string',
         'ff_score',
         'ff_terms',
         'upw_free_fulltext_url',
         'upw_is_boai_license',
         'upw_is_free_to_read',
         'upw_is_subscription_journal',
         'upw_license',
         'upw_oa_color_category',
         'upw_oa_color_verbose',
         'upw_oa_color',  # internal
         'deal_name',
         'deal_owner',
         'deal_discount',
         'deal_discount_verbose',
         'deal_owner_verbose',
         'corresponding_author_surname',
         'match_affiliation_id',
         'match_surname',
         'match_indexed_name',
         'match_auid',
         'match_aut_score',
         'is_corresponding_author_a_vu_author',
         'is_corresponding_author_a_ukb_author']]
        col_rename_dict = {  'publicationName' : 'journal_name',
         'first_affil_author' : 'first_VU_author',
         'first_affil_author_org' : 'first_VU_author_raw_organization_info',
         'ff_match': 'faculty_(matched)',
         'ff_match_subgroup': 'subgroup_(matched)',
         'ff_message': 'diagnostics: ff message',
         'ff_provided_organization_string': 'diagnostics: ff raw input ',
         'ff_score': 'diagnostics: ff score',
         'ff_terms': 'diagnostics: ff matching words',
         'upw_free_fulltext_url': 'fulltext_free_url',
         'upw_is_boai_license': 'is_a_boai_license',
         'upw_is_free_to_read': 'is_free_to_read',
         'upw_is_subscription_journal': 'is_a_subscription_journal',
         'upw_license': 'license',
         #'upw_oa_color_category': '', # internal
         'upw_oa_color_verbose': 'open_access_color',
         #'deal_name',
         'deal_owner' : 'deal_owner_raw',
        # 'deal_discount_verbose', # internal
         'deal_owner_verbose' : 'deal_scope',
         #'corresponding_author_surname',
         'match_affiliation_id' : 'corresponding_author_affiliation_id_(matched)',
         'match_surname': 'corresponding_author_surname_(matched)',
         'match_indexed_name': 'corresponding_author_indexed_name_(matched)',
         'match_auid': 'corresponding_author_author_id_(matched)',
         'match_aut_score': 'diagnostics:corresponding_author_match_score'}
        # 'is_corresponding_author_a_vu_author',
        # 'is_corresponding_author_a_ukb_author'}
        df2 = df2.rename(columns=col_rename_dict)

        def get_contact_point(row):
            if row.is_corresponding_author_a_vu_author is True:
                res = row['corresponding_author_indexed_name_(matched)']
            else:
                res = row['first_VU_author']
            # bij een workflow moet er even op PURE gekeken worden naar de huidige faculteit/groep van de auteur (evt hand/automatisch)
            return res
        df2['vu_contact_person'] = df2.apply(get_contact_point,axis=1)
        if do_save:
            df2.to_csv(path_out + 'knip_OA_VU' + str(MY_YEAR) + '_met_corresponding_authors.csv')
            df2.to_excel(path_out + 'knip_OA_VU' + str(MY_YEAR) + '_met_corresponding_authors.xlsx')

        dict_output[MY_YEAR] = df2

    print('done with scopus arm')
    return dict_output
