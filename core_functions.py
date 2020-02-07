# Core functions
#
# this file contains reusable core functions like filtering on university and adding year and month name info
# these are functions which are generally used in every product


from nlp_functions import remove_punctuation
from nlp_functions import get_abstract_if_any
from nlp_functions import comma_space_fix
from static import PATH_START, PATH_START_PERSONAL
from static import PATH_START_SERVER , PATH_START_PERSONAL_SERVER
from static import UNPAYWALL_EMAIL
from static import PATH_STATIC_RESPONSES
from static import PATH_STATIC_RESPONSES_ALTMETRIC
from static import PATH_STATIC_RESPONSES_SCOPUS_ABS
from static import MAX_NUM_WORKERS  # not used everywhere so care
import pandas as pd
import calendar
import numpy as np
import requests
from pybliometrics.scopus import ScopusSearch
from pybliometrics.scopus import AbstractRetrieval
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import time
from datetime import datetime  # new
from datetime import timedelta
import re
import mysql.connector
from mysql.connector import Error
from altmetric import Altmetric
import pickle
import functools
from unittest.mock import Mock
from requests.models import Response
#import sys
from nlp_functions import faculty_finder


def make_doi_list_from_csv(source_path, output_path, do_return=True):
    # this function returns a list of DOIs from a source scopus frontend file
    # in: source_path: a full path ending with .csv which contains a csv which has a column 'DOI'
    #     output_path: a full path ending with .csv which will be where the result is returned as csv
    # out: a csv is generated and saved, and is returned as dataframe as well
    #
    df = pd.read_csv(source_path)
    df[~df.DOI.isnull()].DOI.to_csv(output_path, header=False)
    if do_return:
        return df[~df.DOI.isnull()].DOI
    else:
        return None


def filter_on_uni(df_in, affiliation_column, cur_uni, affiliation_dict_basic):
    """" returns the dataframe filtered on the chosen university
    in: df with column 'Scopus affiliation IDs' with list of affiliation ids in scopus style
        cur_uni: a university name appearing in the dictionary affiliation_dict_basic
        affiliation_dict_basic: a dictionary with keys unis and values affiliation ids
    out: df filtered over rows
    """

    # now the return has all info per university
    return df_in[df_in.apply(lambda x: not (set(x[affiliation_column].split(', '))
                                            .isdisjoint(set(affiliation_dict_basic[cur_uni]))), axis=1)]


def add_year_and_month_old(df_in, date_col):
    """" adds two columns to a dataframe: a year and a month
    in: df_in: dataframe with special column (read below)
        date_col: name of column which has data information, formatted as [start]YYYY[any 1 char]MM[anything][end]
                  column must not have Nones or nans for example
    out: dataframe with extra columns for year and month
    """

    df_in['year'] = df_in[date_col].apply(lambda x: x[0:4])
    df_in['month'] = df_in[date_col].apply(lambda x: x[5:7])
    df_in['month_since_2018'] = df_in.month.astype('int') + (df_in.year.astype('int')-2018)*12
    df_in['month_name'] = df_in.month.astype('int').apply(lambda x: calendar.month_name[x])

    return df_in


def add_year_and_month(df_in, date_col):
    """" adds two columns to a dataframe: a year and a month
    in: df_in: dataframe with special column (read below)
        date_col: name of column which has data information, formatted as [start]YYYY[any 1 char]MM[anything][end]
                  column must not have Nones or nans for example
    out: dataframe with extra columns for year and month
    """

    df_in['year'] = df_in[date_col].apply(lambda x: None if x is None else x[0:4])
    df_in['month'] = df_in[date_col].apply(lambda x: None if x is None else x[5:7])
    df_in['month_since_2018'] = df_in.apply(lambda x: None if x.month is None else int(x.month) + (int(x.year)-2018)*12, axis=1)
    #df_in.month.astype('int') + (df_in.year.astype('int')-2018)*12
    df_in['month_name'] = df_in.month.apply(lambda x: None if x is None else calendar.month_name[int(x)])

    return df_in


def add_pure_year(df_in, date_col='Current publication status > Date'):
    """" adds one columns to a dataframe: a 'pure_year' based on pure info.
    The input must fit the PURE form as 'Anything+YY'
    We assume the year is after 2000! there are no checks for this
    in: df_in: dataframe with special column (read below)
        date_col: name of column which has data information, formatted as [start][anything]YYYY[end]
                  column must not have Nones or nans for example
    out: dataframe with extra columns for year and month
    """
    if date_col is None:
        df_in['pure_year'] = np.nan
    else:
        df_in['pure_year'] = df_in[date_col].apply(lambda x: float('20' + x[-2:]))

    return df_in


def get_scopus_abstract_info(paper_eid):
    """
    Returns the users df_in with extra columns with scopus abstract info per row or with diagnostics
    :param df_in: must have doi and eid
    :return:
    """

    # init
    no_author_group = True  # we want this too
    error = False
    ab = None
    error_message = 'no error'


    if paper_eid == None:
        # paper_without eid
        error_message = 'paper eid is none'
        error = True
    else:
        try:
            ab = AbstractRetrieval(identifier=paper_eid, view='FULL', refresh=True, id_type='eid')
        except:
            error = True
            error_message = 'abstract api error'

        if not(error):
            # chk if API errors out on authorgroup call and log it
            try:
                ab.authorgroup
                no_author_group = False
            except:
                no_author_group = True

            ##### this belongs in another function, with its own diagnostics + only run ff if this succeeds in topfn
            ####if not(no_author_group):
            ####   (bool_got_vu_author, a, b) = find_first_vu_author()  # yet to make this

            # also if no error, save the result for returns

    return {'abstract_object': ab,
            'no_author_group_warning': no_author_group,
            'abstract_error': error,
            'abstract_error_message': error_message}


def split_scopus_subquery_affils(subquery_affils, number_of_splits=4,
                                 subquery_time = ''):
    """
    ! This function needs testing
    
    This function takes in subquery_affils from make_affiliation_dicts_afids()
    and translates it into a list of subqueries to avoid query length limits
    
    in: subquery_affils from make_affiliation_dicts_afids()
        number_of_splits: an integer between 2 and 10
        subquery_time: an optional query to paste after every subquery
        
    out: a list of subqueries to constrain scopussearch to a subset of affils
         during stacking be sure to de-duplicate (recommended on EID)
         
    """
    if (number_of_splits <= 10) & (number_of_splits > 1) & (number_of_splits % 1 == 0):
        pass  # valid number_of_splits
        # you do not have to worry about number_of_splits < #afids because
        # in python asking indices range outside indices range yields empty lists
        # s.t. stacking them here does nothing
        # needs checking though
    else:
        print('invalid number_of_splits, replacing with 4')
        number_of_splits = 4
    
    affil_count = len(subquery_affils.split('OR'))  # number of affiliation ids
    if affil_count <= 12:  # to avoid weird situations
        print('affil_count is small, returning single subquery')
        my_query_set = subquery_affils + subquery_time
    else: 
        # do it
        my_query_set = []
        step_size = int(np.floor(affil_count / number_of_splits)+1)
        counter = 0
        for cur_step in np.arange(0,number_of_splits):
            if counter == 0:
                cur_subquery = 'OR'.join(subquery_affils.split('OR')[0:step_size]) + ' ) '
            elif counter == number_of_splits-1:  # this is the last one
                cur_subquery = ' ( ' + 'OR'.join(subquery_affils.split('OR')[step_size*cur_step:step_size*(cur_step+1)]) # + ' ) ) '
            else:
                cur_subquery = ' ( ' + 'OR'.join(subquery_affils.split('OR')[step_size*cur_step:step_size*(cur_step+1)])  + ' ) '
            # stack results in a list, check if we need extra [] or not !
            cur_subquery = cur_subquery + subquery_time
            my_query_set.append(cur_subquery)
            counter = counter + 1  # useless but OK
    #print('-----')
    #print(my_query_set)
    #print('-----')

    return my_query_set


def get_first_chosen_affiliation_author(ab, chosen_affid):
    """

    :param ab:
    :return:
    """

    # init
    first_vu_author = None
    cur_org = None
    has_error = False
    first_vu_author_position = None  # care reverse!!! you need a length here or extra unreverse

    try:
        # loop over the authors in the author group, back to front, s.t. the 'first' vu author overwrites everything
        # this is not ideal,
        # because we would also want to check the second vu-author if first one can't be traced back to a faculty
        for cntr, author in enumerate(ab.authorgroup[::-1]):  # ensures the final vu_author result is the leading vu author
            if author.affiliation_id == None:
                # then we can't match as vu author (yet), so we just skip as we do non-vu authors
                1
            else:
                if not (set(author.affiliation_id.split(', ')).isdisjoint(set(chosen_affid))):
                    cur_org = author.organization
                    if author.given_name == None:
                        author_given_name = '?'
                    else:
                        author_given_name = author.given_name
                    if author.surname == None:
                        author_surname = '?'
                    else:
                        author_surname = author.surname
                    first_vu_author = author_given_name + ' ' + author_surname
    except:
        has_error = True

    return {'first_affil_author': first_vu_author,
            'first_affil_author_org': cur_org,
            'first_affil_author_has_error': has_error}


def get_count_of_chosen_affiliation_authors(ab, chosen_affid):
    """

    :param ab:
    :return:
    """

    # init
    author_count_valid = False
    author_count = 0
    has_error = False

    try:
        # loop over the authors in the author group, back to front, s.t. the 'first' vu author overwrites everything
        # this is not ideal,
        # because we would also want to check the second vu-author if first one can't be traced back to a faculty
        for cntr, author in enumerate(ab.authorgroup[::-1]):  # ensures the final vu_author result is the leading vu author
            if author.affiliation_id == None:
                # then we can't match as vu author (yet), so we just skip as we do non-vu authors
                1
            else:
                if not (set(author.affiliation_id.split(', ')).isdisjoint(set(chosen_affid))):
                    # then we have a vu-author. Count and continue
                    # notice there is no safety net if an author appears multiple times for some reason
                    author_count = author_count + 1
                    author_count_valid = True
    except:
        has_error = True
        # then the author_count_valid remains False

    return {'affil_author_count': author_count,
            'affil_author_count_valid': author_count_valid,
            'affil_author_count_has_error': has_error}

# upw start

## 1st at bottom

## 2nd
# remember, these are not for general purpose, but specific decorators for api-harvester-type functions crystal_()
def check_id_validity(func):
    # first layer is a pass right now and that is OK
    def decorator_check_id_validity(func):
        @functools.wraps(func)
        def wrapper_check_id_validity(cur_id, my_requests):
            #
            # pre-process
            valid_doi_probably = False
            if cur_id is not None:
                if pd.notnull(cur_id):
                    if cur_id != 'nan':
                        try:
                            cur_id = cur_id.lower()
                            valid_doi_probably = True
                        except:
                            try:
                                cur_id = str(cur_id).lower()  # not sure but OK
                                valid_doi_probably = True  # stay on safe side then and loose tiny bit of performance
                            except:
                                # then give up
                                print('warning: failed to str(cur_doi).lower()')
            if not valid_doi_probably:
                # chance cur_id s.t. the crystal function can skip the checks and directly insert invalid-id-result
                cur_id = 'invalid'  # the only change
            # end of pre-process
            #
            # run the core function
            r, relevant_keys, cur_id_lower, prepend, id_type = func(cur_id, my_requests)
            #
            # no post-process
            #
            return r, relevant_keys, cur_id_lower, prepend, id_type
        return wrapper_check_id_validity
    return decorator_check_id_validity(func)

#############################################add_deal_info

## 3rd
def check_errors_and_parse_outputs(func):
    # first layer is a pass right now and that is OK
    def decorator_check_errors_and_parse_outputs(func):
        @functools.wraps(func)
        def wrapper_check_errors_and_parse_outputs(cur_id, my_requests):
            #
            # pre-processing
            #
            #
            r, relevant_keys, cur_id_lower, prepend, id_type = func(cur_id, my_requests)
            #
            # post-processing
            #
            # init a dict and fill with right keys and zeros
            dict_init = {}  # values are filled with None as starting point
            for key in relevant_keys:
                dict_init[prepend + key] = None  # really init empty and stays empty if error
            dict_init[prepend + id_type] = None  # can only be data['doi'] (!)  # legacy
            dict_init[prepend + id_type + '_lowercase'] = cur_id_lower
            dict_init['own_' + id_type + '_lowercase'] = cur_id_lower
            dict_init['orig_' + id_type] = cur_id  # legacy
            #
            dict_to_add = dict_init

            # ! somehow need to recognize doi_lowercase too...
            #
            try:
                if 'error' in r.json().keys():
                    # the following code has been checked to work as intended
                    has_error = True
                    error_message = r.json()['message']
                    dict_to_add[prepend + 'error'] = has_error
                    dict_to_add[prepend + 'error_message'] = error_message
                    #
                else:
                    # case: no error
                    #print(r)
                    #print(r.json())
                    has_error = False
                    error_message = 'no error'
                    dict_to_add[prepend + 'error'] = has_error
                    dict_to_add[prepend + 'error_message'] = error_message
                    #
                    # get data
                    try:
                        data = r.json()['results'][0]
                    except:
                        data = r.json()
                    # overwrite dict_to_add with data
                    for key in relevant_keys:
                        try:
                            dict_to_add[prepend + key] = data[key]  # even upw_doi goes automatically : )
                        except KeyError:
                            dict_to_add[prepend + key] = None  # if the key is not there, the result is None
                    dict_to_add[prepend + id_type] = cur_id  # fix

            except:
                has_error = True
                error_message = "error in r.json() or deeper"
                dict_to_add[prepend + 'error'] = has_error
                dict_to_add[prepend + 'error_message'] = error_message
            #
            return pd.Series(dict_to_add)  # r, relevant_keys  # different output  # output has been changed
        return wrapper_check_errors_and_parse_outputs
    return decorator_check_errors_and_parse_outputs(func)




#############################################


## 4th
def faster(func):
    # makes stuff for lists of ids and enables multi-threading and persistent sessions : )  amazing
    # first layer is a pass right now and that is OK
    def decorator_iterate_list(func):
        @functools.wraps(func)
        def wrapper_iterate_list(doi_list, silent=True, multi_thread=True, my_requests=None, allow_session_creation=True):
            """ returns unpaywall info for a given doi list, includes result success/failure and diagnostics
            :param doi_list: doi list as a list of strings, re-computes if doi are duplicate
                             does not de-dupe or dropna for generality, but you can do doi_list = df_in.doi.dropna().unique()
                             if you so desire
                   silent: whether you want silent behaviour or not, defaults to printing nothing
                   multi_thread: whether you want to multi_thread unpaywall (code has been tested), on by default
                                 you do not have to worry about worker counts, a default law is integrated for that
                   my_requests: by default None, but can be exchanged for a requests-session on demand
                                with default, called functions will themselves enter 'requests' to reduce communication costs
                   allow_session_creation: if my_requests=None, this allows the fn to make its own session
            :return: subset of unpaywall columns info + diagnostics as a pandas DataFrame, vertically doi's in lowercase-form.
                     duplicate doi's in the list are ignored, and the output has 1 row per unique DOI

            Notice: this should be the only function to call fn_get_upw_info for more than 1 DOI (for developers)
                    , s.t. the multi-threading code can be here without duplicate code
            """
            # all processing
            # empty dataframe
            df_unpaywall = pd.DataFrame()

            if multi_thread:  # valid across session used or not
                max_num_workers = MAX_NUM_WORKERS
                num_workers = np.max(
                    [1, int(np.floor(np.min([max_num_workers, np.floor(float(len(doi_list)) / 4.0)])))])

            if (my_requests is None) & (allow_session_creation is True) & (len(doi_list) >= 20):
                # then optionally make your own session # + avoid overhead for small jobs
                # perform with a session
                with requests.Session() as sessionA:
                    if multi_thread:
                        fn_get_upw_info_partial = partial(func,
                                                          my_requests=sessionA)  # avoid communication costs
                        multi_result = multithreading(fn_get_upw_info_partial,
                                                      doi_list,
                                                      num_workers)
                        for cur_series in multi_result:
                            df_unpaywall = df_unpaywall.append(cur_series, ignore_index=True)
                    else:  # single thread
                        for (counter, cur_doi) in enumerate(doi_list):
                            if silent == False:
                                print(
                                    'unpaywall busy with number ' + str(counter + 1) + ' out of ' + str(len(doi_list)))
                            cur_res = func(cur_doi, my_requests=sessionA)
                            df_unpaywall = df_unpaywall.append(cur_res, ignore_index=True)
            else:
                # perform without a session
                if multi_thread:
                    fn_get_upw_info_partial = partial(func,
                                                      my_requests=my_requests)  # avoid communication costs
                    multi_result = multithreading(fn_get_upw_info_partial,
                                                  doi_list,
                                                  num_workers)
                    for cur_series in multi_result:
                        df_unpaywall = df_unpaywall.append(cur_series, ignore_index=True)
                else:  # single thread
                    for (counter, cur_doi) in enumerate(doi_list):
                        if silent == False:
                            print('unpaywall busy with number ' + str(counter + 1) + ' out of ' + str(len(doi_list)))
                        cur_res = func(cur_doi, my_requests=my_requests)
                        df_unpaywall = df_unpaywall.append(cur_res, ignore_index=True)

            # either way, return the result
            return df_unpaywall
        return wrapper_iterate_list
    return decorator_iterate_list(func)


## 5th
def appender(func, cur_id_name='doi'):
    """
    Returns the given dataframe with extra columns with unpaywall info and result success/failure and diagnostics
    Merging is done with lower-cased DOI's to avoid duplicate issues. The DOI name is case-insensitive
    :param df_in: df_in as a pandas dataframe, must have a column named 'doi' with doi's as string
    :return: pandas dataframe with extra columns with subset of unpaywall info and result success/failure and diagnostic
             all new doi info is lowercase
    """
    def decorator_appender(func):
        @functools.wraps(func)
        def wrapper_appender(df_in, silent=True, cut_dupes=False, avoid_double_work=True,
                          multi_thread=True, my_requests=None, allow_session_creation=True):


            if cur_id_name == 'eid':
                print('warning: scopus abstract accelerator has not been validated yet !')



            # make doi_list
            if avoid_double_work:
                doi_list = df_in.drop_duplicates(cur_id_name)[cur_id_name].to_list()  # notice no dropna to keep functionality the same
                # also no lower-dropna for simplicity
            else:
                doi_list = df_in[cur_id_name].to_list()

            if cut_dupes:
                print('deprecated code running')
                # I think it should yield exactly the same result, but needs testing that is all
                # overwrites
                doi_list = df_in[cur_id_name].dropna().unique()

            # get unpaywall info
            df_unpaywall = func(doi_list, silent, multi_thread, my_requests, allow_session_creation)

            # merge to add columns
            # prepare doi_lower
            df_in.loc[:, 'id_lowercase'] = df_in[cur_id_name].str.lower()
            df_merged = df_in.merge(df_unpaywall.drop_duplicates('own_' + cur_id_name + '_lowercase'),
                                    left_on='id_lowercase', right_on='own_' + cur_id_name + '_lowercase', how='left')
            # drop duplicates in df_unpaywall to avoid having duplicates in the result due repeating DOI's or Nones
            # assumption: all none returns are the exact same

            if not silent:
                print('done with add_unpaywall_columns')

            return df_merged
        return wrapper_appender
    return decorator_appender(func)




#@appender
#@faster
#@check_errors_and_parse_outputs
#@check_id_validity
def crystal_unpaywall(cur_id, my_requests):
    # always use cur_id, my_requests for in and r, relevant_keys for out
    # id is either cur_doi or 'invalid' if invalid

    prepend = 'upw_'
    id_type = 'doi'
    cur_id_lower = cur_id.lower()

    if my_requests is None:
        my_requests = requests  # avoids passing requests around everytime

    relevant_keys = ['free_fulltext_url',
                     'is_boai_license', 'is_free_to_read', 'is_subscription_journal',
                     'license', 'oa_color']  # , 'doi', 'doi_lowercase'  : you get these from callers
    if cur_id == 'invalid':
        # get the invalid-doi-response directly from disk to save time, you can run update_api_statics to update it
        in_file = open(PATH_STATIC_RESPONSES, 'rb')
        r = pickle.load(in_file)
        in_file.close()
    else:
        r = my_requests.get("https://api.unpaywall.org/" + str(cur_id) + "?email=" + UNPAYWALL_EMAIL)  # force string

    return r, relevant_keys, cur_id_lower, prepend, id_type


# recreate the legacy unpaywall functions for now
#
fn_get_upw_info = check_errors_and_parse_outputs(check_id_validity(crystal_unpaywall))
fn_get_all_upw_info = faster(fn_get_upw_info)
add_unpaywall_columns = appender(fn_get_all_upw_info)


@appender
@faster
@check_errors_and_parse_outputs
@check_id_validity
def crystal_altmetric(cur_id, my_requests):
    """
    This is a bit annoying because this returns either None or a dictionary, and not a request object...
    So I will just send requests without the package
    """

    prepend = 'altmetric_'
    id_type = 'doi'
    cur_id_lower = cur_id.lower()

    if my_requests is None:
        my_requests = requests  # avoids passing requests around everytime

    # some settings
    api_ver = 'v1'  # may change in future, so here it is. For api-key re-edit with altmetric package
    api_url = "http://api.altmetric.com/%s/" % api_ver
    url = api_url + 'doi' + "/" + cur_id


    relevant_keys = ['title', 'cited_by_policies_count', 'score']  # OK for now, care some may miss, patch for that !
    # , 'doi', 'doi_lowercase'  : you get these from callers
    if cur_id == 'invalid':
        # get the invalid-doi-response directly from disk to save time, you can run update_api_statics to update it
        in_file = open(PATH_STATIC_RESPONSES_ALTMETRIC, 'rb')
        r = pickle.load(in_file)
        in_file.close()
    else:
        # r = my_requests.get("https://api.unpaywall.org/" + str(cur_id) + "?email=" + UNPAYWALL_EMAIL)  # force string
        r = my_requests.get(url, params={}, headers={})

    return r, relevant_keys, cur_id_lower, prepend, id_type
add_altmetric_columns = crystal_altmetric  # test me



###@appender(cur_id_name='eid')
@faster
@check_errors_and_parse_outputs
@check_id_validity
def crystal_scopus_abstract(cur_id, my_requests):
    """
    This is a bit annoying because this returns either None or a dictionary, and not a request object...
    So I will just send requests without the package
    """

    prepend = 'scopus_abstract_'
    id_type = 'eid'
    cur_id_lower = cur_id.lower()  # irrelevant but OK

    ### not used
    ###if my_requests is None:
    ####    my_requests = requests  # avoids passing requests around everytime

    # some settings
    # None

    # the issue is that ab is not a requests-type
    # but we need requests-type
    # also, I do not want to use homebrew request code for it because scopus apis are an outsourced mess
    # instead we will use a mock

    relevant_keys = ['obje', 'retries']  # all in one, care integration
    # , 'doi', 'doi_lowercase'  : you get these from callers
    if cur_id == 'invalid':
        # get the invalid-doi-response directly from disk to save time, you can run update_api_statics to update it
        in_file = open(PATH_STATIC_RESPONSES_SCOPUS_ABS, 'rb')
        r = pickle.load(in_file)
        in_file.close()
    else:
        # r = my_requests.get("https://api.unpaywall.org/" + str(cur_id) + "?email=" + UNPAYWALL_EMAIL)  # force string
        # r = my_requests.get(url, params={}, headers={})
        #
        # scopus api is not friendly so I need a try/except here
        #


        # wait-and-retry
        one_shot = False
        if one_shot:
            retries = 0
            try:
                ab = AbstractRetrieval(identifier=cur_id, view='FULL', refresh=True, id_type='eid')
                r = Mock(spec=Response)
                r.json.return_value = {'obje': pickle.dumps(ab), 'message': 'hi', 'retries':retries}
                r.status_code = 999
                # requirements:
                # r.json().keys
                # r.json()['message']
                # r.json()['results']  # if not present, will not unpack and use json().keys()
            except:
                # if so, fall back to invalid routine
                #
                # get the invalid-doi-response directly from disk to save time, you can run update_api_statics to update it
                in_file = open(PATH_STATIC_RESPONSES_SCOPUS_ABS, 'rb')
                r = pickle.load(in_file)
                in_file.close()
        else:
            # print(one_shot)
            retry = True
            retries = -1
            while retry:
                #retry = False  # removes retries
                retries = retries + 1
                try:
                    ab = AbstractRetrieval(identifier=cur_id, view='FULL', refresh=True, id_type='eid')
                    qq = ab.title
                    qqx = qq + 'x'
                    #
                    # if api does not error, and we have an title, then the call is correct and we got info back successfully
                    #
                    # then do rest of actions
                    r = Mock(spec=Response)
                    r.json.return_value = {'obje': pickle.dumps(ab), 'message': 'hi', 'retries': retries}
                    r.status_code = 999
                    retry = False
                except:
                    # we had an api error or a return with empty information
                    # either way, just fillna and continue
                    if retries < 30:
                        retry = True
                        time.sleep(1)
                        if retries > 2:
                            print('retrying ' + str(retries))
                    else:
                        retry = False
                        # prepare for exit
                        in_file = open(PATH_STATIC_RESPONSES_SCOPUS_ABS, 'rb')
                        r = pickle.load(in_file)
                        in_file.close()







        # you have to validate this code because scopus has weird features going in which mess up data when overloading

    return r, relevant_keys, cur_id_lower, prepend, id_type
crystal_scopus_abstract = appender(func=crystal_scopus_abstract, cur_id_name='eid')


class api_extractor:
    """

    DEPRECATED: please stop using this... I will make a new one later, for now updates and patches are stopped

    This class is an api extractor: it extracts info across api's.
    Has multi-threading :)
    Is not an eager operator so ScopusSearch query is only executed when needed and not on initialization
    source_list: which sources to use, like unpaywall
    query: query to put in scopussearch
    Under construction: only does unpaywall data right now to test multi-threading
    Also, I need an extra step for scopussearch datacleaning split-off

    Dubbel-check ff of je de juiste funccorresponding_author_functionsties hebt, bv voor unpaywall drop_dupe stap bij merge
    Plan nu: ff scopussearch-bypass erin, daarmee ff doortesten speedgain op grotere volumes
    """

    def __init__(self,
                 query='TITLE(DATA SCIENCE) AND PUBDATETXT(February 2018)',
                 source_list=['all'],
                 max_num_workers=32):
        self.source_list = source_list
        self.query = query
        self.scopus_search_info = None
        self.scopus_search_info_ready = False
        self.max_num_workers = max_num_workers

    def get_scopus_search_info(self, cur_query):   
        """
        Gets the scopus search info and return it as dataframe of obj.results
        Not yet handling errors of API...
        """

        use_sleep_and_retry = True
        if use_sleep_and_retry:
            no_res = True
            cntr=0
            while no_res:
                try:
                    res = pd.DataFrame(ScopusSearch(cur_query, refresh=True).results)
                    no_res = False
                except:
                    cntr = cntr + 1
                    print(str(cntr) + ' ' + cur_query)
                    time.sleep(1)

        else:
            res = pd.DataFrame(ScopusSearch(cur_query, refresh=True).results)



        return res



    def feed_scopus_search_info(self, df_in, do_return=False, do_overwrite=False):
        """
        This methods allows you to directly feed in a dataframe with scopussearch info,
        of the form pandas.DataFrame(ScopusSearch().results)
        """
        if (self.scopus_search_info_ready is False) | do_overwrite is True:
            self.scopus_search_info = df_in
            self.scopus_search_info_ready = True
            if do_return:
                return self.scopus_search_info
        else:
            print('scopus search info not updated because info was already entered and do_overwrite was provided False')

    def extract(self, use_multi_thread=True, skip_scopus_search=False, skip_unpaywall=False,
                use_parallel_apis=False):
        """
        extract all chosen info
        """
        # the functions like get_scopus_search_info and fn_get_upw_info,
        # should always be single-thread in themselves,
        # and we make them multi-thread outside of their own functions
        #
        # !!! we can further speed up by requesting from api providers in parallel
        # that way we can further avoid api rate limits
        # for this we need advanced functionality
        # after writing the code, turn the default use_parallel_apis to True
        #
        #
        # always redo scopus-search unless explicitly asked skip_scopus_search


        # init
        if not(self.scopus_search_info is None):
            df_temp = self.scopus_search_info.copy()
            doi_list = df_temp[~df_temp.DOI.isnull()].DOI.drop_duplicates().to_list()
            #
            # doi list issue happens here and in getupwdata line 161: search to_list, and doi/DOI difference




            # here: add fn (read jupyter)

        df_upw = pd.DataFrame()
        df_ss = pd.DataFrame()
                
        if use_multi_thread:           
            #ss
            if skip_scopus_search is False:
                # !!! please thoroughly test this
                print('untested functionality called: multithread scopus search: careful!')  # see fast_scopus_search_test.py for dev!
                my_query = self.query  # use own query
                mini_queries = split_query_to_months(my_query)
                count_queries = len(mini_queries)
                # num_workers law: PLEASE TEST IT for optimum point or not
                num_workers = np.max([1, int(np.floor(np.min([self.max_num_workers, np.floor(float(count_queries)/4.0)])))])
                #
                multi_result = multithreading(self.get_scopus_search_info, mini_queries, num_workers)
                for cur_series in multi_result:
                    # we are appending dataframes, not series
                    df_ss = df_ss.append(cur_series, ignore_index=True)
                    ###doi_list = df_ss.doi  # check this !
                    
                    
            ## This is the point where parallel-api functionality should start(!)
            if use_parallel_apis:
                1
                # please first make the apis work in single_thread
                # then in regular multi-thread
                # and finally in parallel_apis_multi_thread.
            
                # 1. set sources using the skip_ arguments
                # 2. choose max_workers using not on #dois but #dois*doi-apis + #eids*eid-apis
                # 3. make a list with 1 element per job, including all details like
                #    [ [doi_1,'unpaywall'], [doi_1,'unpaywall'], [eid_1,'scival']. ...]
                # 4. push that into multi-threading, but use a different function
                #    use the function I started below named get_parallel_api_info()
                #    this function picks up the source in element2 in a list element and 
                #    directs to the right api function
                #    this makes the code superclean to support all forms of threading
                #    while keeping full functionality
                #    also, it needs to add a column with 'source' for differentiation
                # 5. follow the unpaywall code below and append and done
                # 6. for proper testing, split by source column back into df_upw/etc/etc
                #    and give the serial_api routine also a combined df for comparability
                # 7. do extensive testing
                # 8. do timing: how large is the speed gain quantitatively?
                #    this is probably best to test on high-end of very-high-end machines
                #    because we need to hit the api rate limits with serial_apis to see an effect
            
            else:
                #upw
                if skip_unpaywall is False:
                    num_workers = np.max([1, int(np.floor(np.min([self.max_num_workers, np.floor(float(len(doi_list))/4.0)])))])
                    multi_result = multithreading(fn_get_upw_info, doi_list, num_workers)
                    for cur_series in multi_result:
                        df_upw = df_upw.append(cur_series, ignore_index=True)
                #if ~skip_scival:
                #    1
            
            
            
        else:
            # single-thread
            
            # ss
            if skip_scopus_search is False:
                # query fed separately btw
                # 2 lines for clarity for now
                scopus_search_results = self.get_scopus_search_info(self.query)  # care
                self.feed_scopus_search_info(scopus_search_results)  # store in properties
                df_ss = scopus_search_results  # combining results is trivial for single-thread
                ###doi_list = df_ss.doi  # check this !
                
            # upw
            if skip_unpaywall is False:
                for cur_doi in doi_list:
                    series_to_add = fn_get_upw_info(cur_doi)
                    df_upw = df_upw.append(series_to_add, ignore_index=True)
        

        
            
        # scopussearch: the save and .self are issue for multithread, incl
        # overwrite of results properties
        # you need to fix that
        # also, the num_workers law: you need to decide that differently too
        # you prolly have 1 - 120 months, and 1 workers does 1 month a time
        # so you need like #months/3 or a comparable version of the law below
            


        return df_upw, df_ss  # ! merge or combine or store properly later

    def get_parallel_api_info(self, cur_id, source):
        # please check if the multi-threader unpacks list elements, if so use 1 argument
        # and unpack within the function to id/source

        # to distinguish later, add the source as a column (is per DOI/EID)
        source_dict = {'api_source' : source }

        if source == 'unpaywall':
            series_to_add = fn_get_upw_info(cur_id)  # cur_id:cur_doi here
            
        if source == 'scival':
            1
            
        series_to_add = series_to_add.append(pd.Series(source_dict))

        return series_to_add

    def change_max_num_workers(self, max_num_workers):
        self.max_num_workers = max_num_workers



def split_query_to_months(query, silent=False):
    """

    warning: did not pass testing, some data records may not be retrieved

    This function splits a ScopusSearch query into multiple ones
    It takes a query with year indication, and plits it to 1 query per month
    This in turn allows the multi-threading functions of this import framework
    to reduce the computation time
    Otherwise, you will wait a very long serverside wait time and then get a
    lot of data at once with massive download times and possibly more failures
    input: a valid ScopusSearch query string which ends with exactly:
        PUBYEAR > XXXX AND PUBYEAR < YYYY
        with no other appearance of PUBYEAR text
        and there is at least one valid year
        Also, there should not be any month specification, only complete years
        And incomplete years are not allowed (current year at time of call)
        Also, the pubyear clauses should be extra clauses with ands at top level
        please respect this format as the regex functionality is not perfect
    advanced: the month january is also split up, because it generally is twice as large
              as the other months
    """
    # this code can be improved with regex
    
    # extract years
    final_year = str(int(query.split('PUBYEAR < ')[1]) - 1)
    first_year = str(int(query.split('PUBYEAR > ')[1][0:4]) + 1)
    rest_of_query = query.split('PUBYEAR > ')[0]  # probably ending with ' AND'
    
    # make year list
    years = np.arange(int(first_year), int(final_year)+1)

    # define month abbreviations (can split out later)
    #calendar.month_name[ value between 1 and 12]
    # example: PUBDATETXT(February 2018)
    
    query_parts = []
    for year in years:
        for month_number in np.arange(1,12+1):
            if month_number == 1:
                # january is split again in two by open access y/n
                query_parts.append(rest_of_query 
                                   + 'PUBDATETXT('
                                   + calendar.month_name[month_number]
                                   + ' '
                                   + str(year)
                                   + ')'
                                   + ' AND OPENACCESS(1)')
                                   
                query_parts.append(rest_of_query 
                                   + 'PUBDATETXT('
                                   + calendar.month_name[month_number]
                                   + ' '
                                   + str(year)
                                   + ')'
                                   + ' AND OPENACCESS(0)')
            else:                
                query_parts.append(rest_of_query 
                                   + 'PUBDATETXT('
                                   + calendar.month_name[month_number]
                                   + ' '
                                   + str(year)
                                   + ')')
    # careful with using ints and strs together
    
    if ~silent:
        print('query has been split up in ' + str(len(query_parts)) + ' queries for multi-threading')
    return query_parts


def multithreading(func, args,
                   workers):
    with ThreadPoolExecutor(workers) as ex:
        res = ex.map(func, args)
    return list(res)

def multithreading_starmap(func, args,
                   workers):
    with ThreadPoolExecutor(workers) as ex:
        res = ex.starmap(func, args)
    return list(res)

def multiprocessing(func, args,
                    workers):
    with ProcessPoolExecutor(workers) as ex:
        res = ex.map(func, args)
    return list(res)


def my_timestamp():
    # return a sring with current time info
    now = datetime.datetime.now()
    return '_'.join(['', str(now.year), str(now.month), str(now.day), str(now.hour), str(now.minute), str(now.second)])


def add_deal_info(path_deals, path_isn, df_b):
    """
    This function adds columns with deal information to your dataframe
    :param path_deals: path to csv with deals, must have columns: 'ISN':'deal_ISN',
                                                                     'Titel':'deal_journal_title',
                                                                     'Deal naam':'deal_name',
                                                                     'Deal korting':'deal_discount',
                                                                     'Deal type':'deal_owner',
                                                                     'Deal bijgewerkt':'deal_modified',
                                                                     'ISSN':'deal_ISSN'
    :param path_isn: path to csv with table from isn to issn numbers, must have columns ISN and ISSN as translation,

    :param df_b: dataframe with at lesat the columns: issn, eIssn, upw_oa_color
    The parameters should not have any columns matching the names of columns the function is trying to add
    :return: your input dataframe df_b with extra columns
    """
    # load in data from apc deals and isn-issn translation table
    # apc deals
    df_d_base = pd.read_csv(path_deals)
    # isn issn translation table
    df_t = pd.read_csv(path_isn)

    # cleaning
    df_b.at[df_b[df_b.issn.apply(lambda x: True if isinstance(x, list) else False)].index.tolist(), 'issn'] = None

    # now translate isn<>issn
    df_d = df_d_base.merge(df_t, left_on='ISN', right_on='ISN', how='left')
    # rename columns for readability
    df_d = df_d.rename(columns={'ISN': 'deal_ISN',
                                'Titel': 'deal_journal_title',
                                'Deal naam': 'deal_name',
                                'Deal korting': 'deal_discount',
                                'Deal type': 'deal_owner',
                                'Deal bijgewerkt': 'deal_modified',
                                'ISSN': 'deal_ISSN'})
    # remove punctuation in ISSN
    df_d['deal_ISSN_short'] = df_d.deal_ISSN.apply(lambda x: np.nan if x is np.nan else x[0:4] + x[5::])
    # drop deals without ISSN to avoid bad merges (can upgrade later to match on j-names)
    df_d = df_d[~df_d.deal_ISSN.isnull()]

    # merge on both issn and eIssn (extensive exploration show this is safe, see file apcdeals1.ipnyb)
    #
    # complex merge-strategy here with dropping columns
    df_m = df_b.merge(df_d, left_on='issn', right_on='deal_ISSN_short', how='left')
    df_m = df_m.reset_index().rename(columns={'index': 'my_index'})
    cols_d = list(df_d)
    df_m_part_1 = df_m[~df_m.deal_ISSN.isnull()]
    df_m_part_2 = df_m[df_m.deal_ISSN.isnull()].drop(cols_d, axis=1).merge(df_d, left_on='eIssn',
                                                                           right_on='deal_ISSN_short', how='left')
    df_m = df_m_part_1.append(df_m_part_2)
    df_m = df_m.sort_values('my_index').reset_index().drop(['index', 'my_index'], axis=1)
    #
    # give nans some intuition
    df_m['deal_discount_verbose'] = df_m['deal_discount'].apply(lambda x: 'no known deal' if x is np.nan else x)
    # df_m['upw_oa_color_verbose'] = df_m['upw_oa_color'].apply(lambda x: 'unknown' if x is np.nan else x)  # wrongplace
    df_m['deal_owner_verbose'] = df_m['deal_owner'].apply(lambda x: 'no known deal' if x is np.nan else x)

    return df_m


def pre_process_pure_data(df,
                          org_info,
                          path_to_save=None,
                          test_mode_upw=False,
                          do_save=False,
                          silent=False):
    """
    Sorry for documentation, time lines are tight.
    This goes in:
    df = dataframe from pure, conditions are tight (will follow)
    org_info is an excel with 2 columns, 1 'Naam' and 1 'Faculteit' which map groups to faculties
    path_to_save: path to where to save as string
    test_mode_upw: whether you want to do unpaywall load for first few records or all of them
    do_save: whether you want to save or not
    This comes out:
    the cleaned, preprocesses dataframe with unpaywall
    """

    # clean column numbering first
    df.columns = [re.sub('^\d+.', "", x) for x in
                       df.columns]  # remove at start of string where 1 or more digits
    df.columns = [re.sub('^\d+', "", x) for x in df.columns]
    df.columns = [re.sub('^ ', "", x) for x in df.columns]
    df.columns = [re.sub('^.\d+', "", x) for x in df.columns]
    df.columns = [re.sub('^ ', "", x) for x in df.columns]


    # hidden settings
    #
    df = df[[
     'Title of the contribution in original language',
     'Current publication status > Date',
     #'5.1 Publication statuses and dates > E-pub ahead of print[1]',
     'Subtitle of the contribution in original language',  # new
     'Type',
     'Workflow > Step',
     'Original language',
     'Electronic version(s) of this work > DOI (Digital Object Identifier)[1]',
     'Organisations > Organisational unit[1]',
     'Organisations > Organisational unit[2]',
     'Organisations > Organisational unit[3]',
     'Organisations > Organisational unit[4]',
     'Organisations > Organisational unit[5]',
     'Organisations > Organisational unit[6]',
     'Organisations > Organisational unit[7]',
     'Organisations > Organisational unit[8]',
     'Organisations > Organisational unit[9]',
     'Organisations > Organisational unit[10]',
     'Journal > Journal[1]:Titles',
     'Journal > Journal[1]:ISSNs',
    # '14.3 Journal > Journal[1]:Additional searchable ISSN (Electronic)',
     'UUID',
    # '18 Created',
    # "33.1 Keywords in 'Open Access classification'[1]"
    ]]

    admitted_types = ['Chapter in Book / Report / Conference proceeding - Chapter',
           'Contribution to Journal - Article',
           'Contribution to Conference - Paper',
           'Book / Report - Report',
           'Book / Report - Book',
           'Chapter in Book / Report / Conference proceeding - Conference contribution',
           'Contribution to Journal - Review article',
    ]



    # pre-processing
    #
    # some robustness needed... some asserts too
    #
    admitted_types_lower = pd.DataFrame(admitted_types)[0].str.lower().to_list()
    df = df[df['Type'].str.lower().isin(admitted_types_lower)]
    ###df = df[df['Type'].isin(admitted_types)]
    ###df = df[df['Type'].isin(admitted_types)]
    df['DOI'] = df['Electronic version(s) of this work > DOI (Digital Object Identifier)[1]']



    # add unpaywall info
    #
    ae = api_extractor(max_num_workers=16)  # care: not tested sufficiently, may give too many error returns
    if test_mode_upw:
        ae.feed_scopus_search_info(df_in=df.iloc[0:1,:], do_overwrite=True)  # 0:1 saves 15sec wait per year of data
        df_res_upw, _ = ae.extract(use_multi_thread=False, skip_scopus_search=True, skip_unpaywall=False,
                                   use_parallel_apis=False)
    else:
        print('multithread is not used')
        ae.feed_scopus_search_info(df_in=df, do_overwrite=True)
        df_res_upw, _ = ae.extract(use_multi_thread=False, skip_scopus_search=True, skip_unpaywall=False,
                                   use_parallel_apis=False)
    #
    # merge back in with orig_doi no nans
    # cleaning is done in the import framework, saving us work and duplicate code : )
    # ! Not sure if dois in pure have an error, causing a mismatch with scopus and unpaywall
    print(list(df_res_upw))
    print(df_res_upw.head(1))
    df = df.merge(df_res_upw, left_on = 'DOI', right_on = 'orig_doi', how = 'left')
    df['upw_oa_color_verbose'] = df['upw_oa_color'].apply(lambda x: 'unknown' if x is np.nan else x)
    ###df_m['pure_oa_class_verbose'] = df_m["33.1 Keywords in 'Open Access classification'[1]"].apply(lambda x: 'unknown' if x is np.nan else x)

    # add faculty_finder info exploiting pure org columns
    #
    ff = faculty_finder(organizational_chart=org_info)

    #
    #
    if silent is False:
        trysize = 100
        start = time.time()
        df.loc[0:trysize,"Organisations > Organisational unit[1]"].apply(lambda x: ff.match(x))
        end = time.time()
        print(end-start)
        print('that was time for 100 entries, but total df is: ')
        print(len(df))
        print('now doing all of em')
        print('this will probably take ' + str(float(len(df))/trysize*(end-start)) + ' seconds')
    #
    #


    df['ff'] = df.loc[:,"Organisations > Organisational unit[1]"].apply(lambda x: ff.match(x))


    df.loc[:, 'ff_provided_organization_string'] = df.ff.apply(lambda x: x['ff_provided_organization_string'])
    df.loc[:, 'ff_match'] = df.ff.apply(lambda x: x['ff_match'])
    df.loc[:, 'ff_score'] = df.ff.apply(lambda x: x['ff_score'])
    df.loc[:, 'ff_terms'] = df.ff.apply(lambda x: x['ff_terms'])
    df.loc[:, 'ff_message'] = df.ff.apply(lambda x: x['ff_message'])
    df.loc[:, 'ff_match_subgroup'] = df.ff.apply(lambda x: x['ff_match_subgroup'])
    #
    # evaluation is in pure_integratie.ipnyb

    # for completeness, I also want ff_match based on org_info

    # extra processing
    df['DOI_isnull'] = df.DOI.isnull()
    df['pub_uuid'] = df['UUID']

    # now save
    if do_save:
        df.to_csv(path_to_save)

    return df


def get_eid_uuid_data(host, database, user, pw, silent=False):
    """
    This function obtains the EID<>PURE_PUB_UUID table from our extrapure database
    It immediately works for all years at once
    :param host: host database (IP)
    :param database: database name
    :param user: user to log into database with
    :param pw: password to log into database with
    :param silent: whether you want to silence extra prints or not
    :return: 1 a dataframe with 2 columns as EID<>PURE_PUB_UUID table if success, otherwise just None
             2 a boolean which is True iff success otherwise False
    """
    try:
        connection = mysql.connector.connect(host=host,
                                             database=database,
                                             user=user,
                                             password=pw)

        sql_select_Query = "select * from scopus_has_publication"
        cursor = connection.cursor()
        cursor.execute(sql_select_Query)
        records = cursor.fetchall()
        df_t = pd.DataFrame(records).rename(columns={0: 'eid', 1: 'pub_uuid'})
        if silent is False:
            print("Total number of rows is: ", cursor.rowcount)
        success = True
    except Error as e:
        #always print this, later also add logging
        print("Error reading data from MySQL table", e)
        print('returning None')
        df_t = None
        success = False
    finally:
        if (connection.is_connected()):
            connection.close()
            cursor.close()
            if silent is False:
                print("MySQL connection is closed")
    return df_t, success


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


def left_pad(my_str):
    if len(my_str) < 2:
        return '0' + my_str
    else:
        return my_str


def get_today():
    return str(datetime.now().year) + '-' + left_pad(str(datetime.now().month)) + '-' + left_pad(
        str(datetime.now().day))


def get_today_for_pubdatetxt():
    return left_pad(calendar.month_name[datetime.now().month]) + ' ' + str(datetime.now().year)

def get_today_for_pubdatetxt_integers(year, month):
    return left_pad(calendar.month_name[month]) + ' ' + str(year)

def get_today_for_pubdatetxt_super(months_back=0):
    # remove datetime. later
    # dt_obj = datetime.datetime.now() - datetime.timedelta(days=datetime.datetime.now().day)

    if months_back == 0:
        dt_obj = datetime.now()
    else:
        cntr = months_back
        dt_obj = datetime.now()
        while cntr > 0:
            dt_obj = dt_obj - timedelta(days=dt_obj.day)
            #print(dt_obj)
            cntr -= 1

    return left_pad(calendar.month_name[dt_obj.month]) + ' ' + str(dt_obj.year)


def make_types_native_basic(lst):
    res = []
    for ii in lst:
        #print(type(ii))
        if (type(ii) == np.int32) | (type(ii) == np.int64):
            #print('aa')
            res.append(int(ii))
        else:
            res.append(ii)
    return res


def add_abstract_to_scopus(start_path,
                           year,
                           do_save_csv=True):
    """
    Combines the scopus pickle and old scopus csv into a new one with cleaned abstract text

    :param start_path: the starting path where all input/output goes. Subdirectories are required.
                        this function requires the subfolder:
                        - 'scopus_processed' with 'pickle_OA_VU'+year+'_met_corresponding_authors.pkl' and for every year
                          'knip_OA_VU'+year+'_met_corresponding_authors.csv'
    :param do_save_csv: whether you want to output a csv or not (will overwrite)
    :return: Nothing
    """
    #
    # get scopus pkl file
    df_pickle = pd.read_pickle(start_path + r'\scopus_processed\pickle_OA_VU' \
                               + str(year) + '_met_corresponding_authors.pkl')
    # make abstract text and clean it
    df_pickle['abstract_text'] = df_pickle.apply(get_abstract_if_any, axis=1)
    df_pickle['abstract_text_clean'] = (df_pickle['abstract_text']
                                        .apply(comma_space_fix)
                                        .apply(remove_punctuation))
    df_pickle = df_pickle[['eid', 'abstract_text_clean']]
    if ((len(df_pickle[df_pickle.eid.isnull()]) > 0)
            | (df_pickle.eid.apply(lambda x: x is None).max())
            | (df_pickle.eid.apply(lambda x: x == 'None').max())):
        print('merge issue: df_pickle for abstract text has some null eids')
    #
    # read scopus
    df_k = pd.read_csv(start_path + r'\scopus_processed\knip_OA_VU' + str(
        year) + '_met_corresponding_authors.csv')
    #
    # merge with scopus
    df_m = df_k.merge(df_pickle[['eid', 'abstract_text_clean']], on='eid', how='left')
    if len(df_m) != len(df_k):
        print('len messed up')
    #
    # save it
    if do_save_csv:
        df_m.to_csv(start_path + r'\scopus_processed\knip_OA_VU' \
                    + str(year) + '_met_abstract_tekst.csv')

    return None


def merge_pure_with_scopus_data(df_p, df_s, df_t):
    """
    This functions merges a pre-processed Pure dataframe with a pre-processed Scopus dataframe, also uses extrapure
    It is a mega-merge using EID, DOI and title with advanced rule sets. Soft-title-match is not included.
    There is room for improvement: a doi-cleaner would be nice like '10.' start for all entries

    This function is year_range-indifferent and will work with any year-range or period-range

    :param df_p: Dataframe from pure, must be preprocessed with pre_process_pure_data()
    :param df_s: Dataframe from scopus, must be enriched through open-access-pipeline (! not yet in Pycharm !)
    :param df_t: Dataframe from xpure with eid to uuid. Careful with UUID: every PURE repo has different uuids.
    :return: df_combined (the merged dataframe including merge_source), diagnostics (is None right now)
    """

    # 1. use df_t to enrich df_p with eids, continue with df_m
    df_m = df_p.merge(df_t, left_on='pub_uuid', right_on='pub_uuid', how='left')
    df_m['has_eid'] = ~df_m.eid.isnull()
    if len(df_m[df_m['Title of the contribution in original language'].isnull()] > 0):
        print('there were records with empty titles and those were discarded')

    # 2. de-duplicate left=pure and right=scopus
    # 2A. de-dupe for eids
    # assumption: last duplicate entry is correct, rest is false
    # we need to preserve records which have NaNs in their eids
    # plan of attack: split part with eid, de-dupe it w/o worrying about nan eids, then re-append the part w/o eid
    df_m = df_m[df_m.eid.isnull()].append(df_m[~df_m.eid.isnull()].drop_duplicates(subset=['eid'], keep='last'))
    if df_m[~df_m.eid.isnull()].eid.value_counts().max() != 1:
        print('eid de-duplication failed somehow')
    # 2B. de-duplicate on DOI
    #     some are maked as 'do_not_merge_on_DOI' which is an advanced feature
    # assumptions:
    # step 1. all records with double DOI except for books and book chapters: keep=last, drop other records
    # step 2. all records with double DOI and =book or =bookchapter: add a flag to not merge on DOI at all,
    #         keep rest as is so we can unpaywall it later
    #
    # prepare support variables
    doi_counts = df_m[~df_m.DOI.isnull()].DOI.value_counts().sort_values(ascending=False)
    double_doi = doi_counts[doi_counts > 1].index.to_list()  # for future use or to mail Reinout or whatever
    df_m['type_contains_book'] = df_m.Type.str.lower().str.contains('book')
    #
    # step 1: drop some of the DOI duplicates (see step 1/2 disc above)
    df_m = (df_m[(~df_m.DOI.isin(double_doi)) | (df_m.type_contains_book)]
            .append(df_m[(df_m.DOI.isin(double_doi)) & (~df_m.type_contains_book)]
                    .drop_duplicates(subset='DOI', keep='last')))
    #
    # step 2: prepare 'do_not_merge_on_DOI' tag
    df_m['do_not_merge_on_DOI'] = ((df_m.DOI.isin(double_doi)) & (df_m.type_contains_book))
    doi_counts = df_m[~df_m.DOI.isnull()].DOI.value_counts().sort_values(ascending=False)
    double_doi = doi_counts[doi_counts > 1].index.to_list()  # for future use or to mail Reinout or whatever
    if df_m[df_m.DOI.isin(double_doi)].do_not_merge_on_DOI.mean() != 1:
        print('doi de-duplication failed somehow')
    # 2C. de-duplicate on titles
    #
    # drop records where there are more than 1 word in the title (where title duplicate)
    # where there is 1 word in the title, we cannot drop, and we should not merge either, so isolate those
    # like 'introduction' can be title-dupe, of course, and still be a unique article
    #
    # this is a hard choice, but it is probably best to remove dupes and add flags before any merge happens,
    # in order to avoid having dupes with different eids appear twice in merged and unmerged form
    # the total affected records are 0.7% and the chance on a missing merge is even smaller
    # this is an assumption: we assume we the kept dupes are the correct and best ones here
    #
    # helper variables
    df_double_titles = df_m['Title of the contribution in original language'].value_counts()
    double_titles = df_double_titles[df_double_titles > 1].index.to_list()
    #
    # btw: these are exclusive sets, any record can belong to maximally one of these two groups
    df_m['is_dupe_based_on_long_title_dupe'] = (
                (df_m['Title of the contribution in original language'].isin(double_titles))
                & (df_m['Title of the contribution in original language'].str.split().str.len() > 1))
    df_m['do_not_merge_on_title'] = ((df_m['Title of the contribution in original language'].isin(double_titles))
                                     & (df_m[
                                            'Title of the contribution in original language'].str.split().str.len() == 1))
    #
    # now we need to remove dupes
    # split into two, drop dupes, then combine back
    df_m = (df_m[df_m['is_dupe_based_on_long_title_dupe']]
            .drop_duplicates(subset=['Title of the contribution in original language'], keep='last')
            .append(df_m[~df_m['is_dupe_based_on_long_title_dupe']]))
    #
    # end of de-duplication and tagging 'do_not_merge_on_DOI' and 'do_not_merge_on_title'


    # 3. Perform the mega-merge
    #
    # drop where title is empty
    df_m = df_m[~df_m['Title of the contribution in original language'].isnull()]
    if len(df_m[df_m['Title of the contribution in original language'].isnull()]) > 0:
        print('dropped ' + str(len(
            df_m[df_m['Title of the contribution in original language'].isnull()])) + '  records for no title present')
    #
    # all variables of step 1
    #
    # first part of pure with eid
    df_A = df_m[~df_m.eid.isnull()]
    df_BC = df_m[df_m.eid.isnull()]
    #
    # inner-merged part of A and Scopus
    df_Amerged_SA = df_A.merge(df_s, on='eid', how='inner')
    #
    # find out which eids were merged on
    merged_eids = set(df_Amerged_SA.eid.unique())
    # merged parts of left and right
    df_Amerged = df_A[df_A.eid.isin(merged_eids)]
    df_SA = df_s[
        df_s.eid.isin(merged_eids)]  # remember we de-duplicated for eids, dois and titles, therefore this should work
    # unmerged parts left and right
    df_Aunmerged = df_A[~df_A.eid.isin(merged_eids)]
    df_Sunmerged1 = df_s[~df_s.eid.isin(merged_eids)]
    #
    # reflux df_Aunmerged
    df_BC_Aunmerged = df_BC.append(df_Aunmerged)
    #
    # all variables of step 2
    # do respect 'do_not_merge_on_DOI'
    #
    # grab from PURE table the B, the C and the Aunmerged parts only
    # do not grab Amerged because we do not want to merge the merged parts again ever
    # from these parts, isolate the parts which fulfill the two conditions: has DOI and has no flag to not merge on DOI
    # these should be attempted to merge on DOI with Scopus (again, do not merge twice, use Sunmerged1 for this)
    # after the merge can obtain the DOIs that merged and use that to split Bmerged and Bunmerged
    # notice that there is a difference with the initial plan: Bunmerged will not contain do_not_merge_on_DOI-set at all
    # To reduce complexity and adhere to the original plan, we will append the do_not_merge_on_DOI-set to Bunmerged
    #
    # also, df_BC_Aunmerged splits up in 3 parts
    # first we cut off the do_not_merge_on_DOI pat
    # then we cut the rest in two: one part without DOI and one part with DOI
    # this last part is the merge_candidate for step 2/B
    df_merge_candidate_B = df_BC_Aunmerged[(~df_BC_Aunmerged.DOI.isnull()) & (~df_BC_Aunmerged.do_not_merge_on_DOI)]
    df_BC_Aunmerged_wo_DOI_may_merge = df_BC_Aunmerged[
        (df_BC_Aunmerged.DOI.isnull()) & (~df_BC_Aunmerged.do_not_merge_on_DOI)]
    df_do_not_merge_on_DOI = df_BC_Aunmerged[df_BC_Aunmerged.do_not_merge_on_DOI]
    #
    # merge
    # assumption: we assume flat doi merge is perfect (we do not lowercase or clean starts or anything)
    # diagnostics: this merges 15 out of 328 pure entries with DOI
    # lowercasing only affects 20% roughly, but merge stays at 15
    # 8 records in total have start different than '10.'
    # I will leave it as uncleaned doi-merging here because the added value is very small
    df_Bmerged_SB = df_merge_candidate_B.merge(df_Sunmerged1, left_on='DOI', right_on='doi', how='inner')
    #
    # find out which dois were merged on
    merged_dois = set(df_Bmerged_SB.DOI.unique())
    merged_dois
    # merged parts of left and right
    df_Bmerged = df_merge_candidate_B[df_merge_candidate_B.DOI.isin(merged_dois)]
    df_SB = df_Sunmerged1[df_Sunmerged1.doi.isin(merged_dois)]
    # unmerged parts left and right
    df_Bunmerged_temp = df_merge_candidate_B[~df_merge_candidate_B.DOI.isin(merged_dois)]
    df_Sunmerged2 = df_Sunmerged1[~df_Sunmerged1.doi.isin(merged_dois)]
    #
    # append the do_not_merge_on_DOI-set to Bunmerged afterwards
    # remember to add the do_not_merge_on_DOI set to df_Bunmerged
    # notice that defining every part explicitly makes this less difficult
    df_Bunmerged = df_Bunmerged_temp.append(df_do_not_merge_on_DOI)
    #
    # info:
    # in step 2 the unmerged parts together were df_BC_Aunmerged
    # we split that now into:
    # 1. df_do_not_merge_on_DOI
    # 2. df_BC_Aunmerged_wo_DOI_may_merge
    # 3. df_merge_candidate_B, which consists of df_Bmerged and df_Bunmerged_temp
    # Also, df_Bunmerged is basically df_Bunmerged_temp + df_do_not_merge_on_DOI
    #
    # so what will be the unmerged part for the next step then?
    # df_do_not_merge_on_DOI + df_BC_Aunmerged_wo_DOI_may_merge + df_Bunmerged_temp
    # or equivalently:
    # df_Bunmerged + df_BC_Aunmerged_wo_DOI_may_merge
    # or equivalently:
    # the unmerged set of the next step is the unmerged set of this step, minus df_Bmerged because that part merged
    # but we'd rather append than 'substract' so we build it up as (in reflux formulation):
    #
    # unmerged part for the next step = df_BC_Aunmerged_wo_DOI_may_merge + df_Bunmerged
    # verified logically a few times now, let's continue
    #
    # reflux df_Bunmerged
    df_C_Bunmerged = df_BC_Aunmerged_wo_DOI_may_merge.append(df_Bunmerged)
    #
    # all variables of step 3
    # do respect 'do_not_merge_on_title'
    #
    # the unmerged set is exactly df_C_Bunmerged
    # but not everything is merge candidate
    # we have to isolate the do_not_merge_on_title set
    df_do_not_merge_on_title = df_C_Bunmerged[df_C_Bunmerged.do_not_merge_on_title]
    df_merge_candidate_C = df_C_Bunmerged[~df_C_Bunmerged.do_not_merge_on_title]
    # notice that we do not split into whether title is present, because title-less records were discarded (0 in 2018)
    #
    # now we have to try to merge on title
    # first we do an exact match merge,
    # for the rest we evaluate the levenshtein distance
    # exploration indicated that we expact very favourable 0/1 splits and no gray zone, but let's try it out
    #
    # first exact match on title
    df_Cmerged_SC_exact = df_merge_candidate_C.merge(df_Sunmerged2,
                                                     left_on='Title of the contribution in original language',
                                                     right_on='title',
                                                     how='inner')
    # now split merged, unmerged and do_not_merge
    # find out which eids were merged on
    merged_titles = set(df_Cmerged_SC_exact.title.unique())
    # merged parts of left and right
    df_Cmerged = df_merge_candidate_C[
        df_merge_candidate_C['Title of the contribution in original language'].isin(merged_titles)]
    df_SC = df_Sunmerged2[df_Sunmerged2.title.isin(merged_titles)]
    # unmerged parts left and right
    df_Cunmerged_temp = df_merge_candidate_C[
        ~df_merge_candidate_C['Title of the contribution in original language'].isin(merged_titles)]
    df_Sunmerged3 = df_Sunmerged2[~df_Sunmerged2.title.isin(merged_titles)]
    # and we have the do_not_merge_on_title set ready, do not forget, better add it now
    df_Cunmerged = df_Cunmerged_temp.append(df_do_not_merge_on_title)
    #
    #
    # This is without soft-title-matching!

    # generate resulting combined table (name it SP)
    # ! careful! you cant just add stuff, we absorbed Aunmerged for example!
    # first append cols to unmerged parts
    df_Amerged_SA.loc[:, 'merge_source'] = 'both'
    df_Bmerged_SB.loc[:, 'merge_source'] = 'both'
    df_Cmerged_SC_exact.loc[:, 'merge_source'] = 'both'
    df_Cunmerged.loc[:, 'merge_source'] = 'pure'
    df_Sunmerged3.loc[:, 'merge_source'] = 'scopus'

    df_combined = (df_Amerged_SA
                   .append(df_Bmerged_SB, sort=False)
                   .append(df_Cmerged_SC_exact, sort=False)
                   .append(df_Cunmerged, sort=False)
                   .append(df_Sunmerged3, sort=False))

    diagnostics = None

    return df_combined, diagnostics


def prepare_combined_data(start_path,
                         year_range=(2017, 2018, 2019),
                         add_abstract=True,
                         skip_preprocessing_pure_instead_load_cache=False,  # safe
                         remove_ultra_rare_class_other=True,
                         path_pw=PATH_START_PERSONAL,
                         org_info=pd.read_excel( PATH_START + r'raw data algemeen\vu_organogram_2.xlsx', skiprows=0)):
    """
    This function prepares the combined data for a chosen year_range
    The raw pure files and processed scopus files per year should be available
    Next step: test this function!

    Remember that you must do a fresh run if you want any different year range !
    In no way can the results be stacked across different executions of this function (including any soft-title-match)
    Because otherwise you will introduce duplicates with that stacking

    :param start_path: the starting path where all input/output goes. Subdirectories are required.
                        this function requires the subfolder:
                        - 'scopus_processed' with 'pickle_OA_VU'+year+'_met_corresponding_authors.pkl' and for every year
                          'knip_OA_VU'+year+'_met_corresponding_authors.csv'
                          in year_range
                        -
    :param year_range:
    :param add_abstract:
    :param remove_ultra_rare_class_other:
    :param skip_preprocessing_pure_instead_load_cache:
    :return:
    """


    # 1. prepare helper variables
    # 1A. wrap immutable parameters
    year_range = list(year_range)
    # 1B. load xpure user/pass
    host = pd.read_csv(path_pw + r'\password_xpure.csv').host[0]
    database = pd.read_csv(path_pw + r'\password_xpure.csv').database[0]
    user = pd.read_csv(path_pw + r'\password_xpure.csv').user[0]
    pw = pd.read_csv(path_pw + r'\password_xpure.csv').pw[0]

    # 2. add abstract
    if add_abstract:
        # add the abstract and set scopus_variant to use this enriched csv
        scopus_variant = '_met_abstract_tekst.csv'
        for year in year_range:
            add_abstract_to_scopus(start_path, year)  # verified: safe for per-year run (scopus<>scopus only)
    else:
        # do not add an abstract and use the original csv
        scopus_variant = '_met_corresponding_authors.csv'

    # 3. Obtain df_combined for a single year
    #    includes obtaining processed pure, scopus and xpure data, then merging it and saving csvs
    df_p_multi_year = pd.DataFrame()
    df_s_multi_year = pd.DataFrame()
    # df_t is always multi-year
    for year in year_range:
        path_pure_unprocessed = start_path + r'\pure_raw\vu' + str(year) + '_public_raw.xls'
        path_scopus = start_path + r'\scopus_processed\knip_OA_VU' + str(year) + scopus_variant
        path_to_save_or_load_processed_pure = start_path + r'\pure_processed\processed_pure' + str(year) + '.csv'

        # 3.1: get processed pure data
        # pre-process the pure data or load a cache
        if skip_preprocessing_pure_instead_load_cache:
            # load processed pure in directly
            df_p = pd.read_csv(path_to_save_or_load_processed_pure)
        else:
            # load in unprocessed pure, process it, save it, read it
            df_p_unprocessed = pd.read_excel(path_pure_unprocessed)
            df_p = pre_process_pure_data(df=df_p_unprocessed,
                                         org_info=org_info,
                                         path_to_save=start_path + r'\pure_processed\processed_pure' + str(year) + '.csv',
                                         test_mode_upw=True,  # True avoids waste since our enriched scopus has it too
                                         do_save=True)  # just always save

        # 3.2: get processed scopus data
        df_s = pd.read_csv(path_scopus)
        #

        # append to stack years
        df_p_multi_year = df_p_multi_year.append(df_p, ignore_index=True)
        df_s_multi_year = df_s_multi_year.append(df_s, ignore_index=True)

    # these parts are multi_year
    # 3.1&3.2 extra: reset indices and append year columns where necessary
    df_p_multi_year = df_p_multi_year.reset_index(drop=True)
    df_s_multi_year = df_s_multi_year.reset_index(drop=True)
    df_s_multi_year['scopus_year'] = df_s_multi_year.year
    if np.min(year_range) >= 2000:
        df_p_multi_year = add_pure_year(df_p_multi_year, date_col='Current publication status > Date')
    else:
        print('violation of the post-2000 assumption for using pure year information')
        df_p_multi_year = add_pure_year(df_p_multi_year, date_col=None)

    # 3.3: get xpure data
    df_t, success_df_t = get_eid_uuid_data(host, database, user, pw, silent=False)

    # 3.4: run the merger for all years at once to avoid the cross-year issue where scopus and pure have different years
    df_combined, diagnostics_merger = merge_pure_with_scopus_data(df_p_multi_year, df_s_multi_year, df_t)


    # 3.5: prepare identifiers for STM to back-merge on... put this higher up please
    df_combined['post_merge_id'] = 1
    df_combined['post_merge_id'] = df_combined['post_merge_id'].cumsum()
    # this post_merge_id also lives forth in df_chosen_year and the unmerged csvs, so you can use it for STM backmerge


    # 4. remove rare classes if desired
    if remove_ultra_rare_class_other:
        df_combined = df_combined[
            df_combined.ff_match != 'VU - Other Units']  # prevent issues with a brand new ultra-rare class please
        # overwrite it
        df_combined.to_csv(start_path + r'\merged_data\df_total.csv')
        df_combined.to_pickle(start_path + r'\merged_data\df_total.pkl')

    # 5: save the full data
    df_combined.to_csv(start_path +
        r'\merged_data\df_total.csv')
    df_combined.to_pickle(start_path +
        r'\merged_data\df_total.pkl')

    # 6. return the verified middle year (which does not suffer from cross-year issue)
    #    Remember that you must do a fresh run if you want any different year range !
    # how do we filter df_combined?
    # P+S and S rows: filter on scopus_year
    # P rows: filter on pure_year
    # this is safe as long as you only do this with a single df_combined for any slice you want
    # why?
    # the df_combined is by assumption duplicate-free. All duplicates of raw-P and raw-S are removed,
    # and then they are merged and again duplicates are removed over the rows.
    # Because all P+S&S rows have exactly only 1 year,
    # and the P rows have exactly only 1 year as well
    # so any proper slice is safe as you don't have anything double if you slice over years and stack again
    # however, you should not involve a second df_combined as the paper may merge in one df_combined and remain
    # unmerged in another df_combined due to a different year_range, and subsequently get a different year to slice on
    # like scopus_year in one and pure_year in another
    # this is an intricate detail, so please avoid such a merge and just re-run or else the data will be dirty and you
    # will not notice at all probably
    for chosen_year in year_range[1:-1]: # drop edges regardless, not checking if last year is last (due future papers!)
        df_chosen_year = (df_combined[(df_combined.merge_source == 'pure')
                                      &
                                      (df_combined.pure_year == chosen_year)]
                          .append(df_combined[(df_combined.merge_source != 'pure')
                                              &
                                              (df_combined.scopus_year == chosen_year)]
                                  )
                          )
        df_chosen_year.to_pickle(start_path + r'\merged_data\df_total' + str(chosen_year) + '.pkl')
        df_chosen_year.to_csv(start_path + r'\merged_data\df_total' + str(chosen_year) + '.csv')

    # 7. isolate unmerged for soft-title-matching: [ notice we do this post-everything to allow early-access-data]
    # df_unmerged = df_combined[(df_combined.merge_source != 'both')]
    # df_unmerged.to_csv(start_path + r'\merged_data\df_unmerged.csv')
    df_unmerged_pure = df_combined[df_combined.merge_source == 'pure']
    df_unmerged_scopus = df_combined[df_combined.merge_source == 'scopus']
    # save to csv
    df_unmerged_pure.to_csv(start_path + r'\df_unmerged_pure.csv')  # equivalent to df_combined/ms=pure
    df_unmerged_scopus.to_csv(start_path + r'\df_unmerged_scopus.csv')  # equivalent to df_combined/ms=scopus

    # 8. you can now run STM with its current settings
    #
    # I am not sure how to deal with the multiprocessing aspect and hardcode entry

    # 1. config and run prepare_combined_data to get triple-merge
    # 2. config and run nlp2
    # 3. config and run incorporate_stm_results

    return None


def get_altmetric(row, col='cited_by_policies_count'):
    """

    legacy code, please use crystal_altmetric() or its aliases instead

    Returns Altmetric's cited_by_policies_count for a given doi
    ! There is no internal cleaning yet
    If DOI is empty or altmetric returns None, then function returns np.nan
    If Altmetric returns non-empty but cited_by_policies_count is missing,
    then the function returns 0
    else returns the cited_by_policies_count
    
    In: DOI and col[functionality missing]
    Out: single value with either np.nan, 0, or a positive integer
    """
    
    if col != 'cited_by_policies_count':
        print('functionality missing for other columns, giving policies now')
        
    if not(pd.notnull(row)):
        out = np.nan
    else:
        a = Altmetric()
        result = a.doi(row)
        if result is None:
            out = np.nan  # not zero!
        else:
            try:
                out = result['cited_by_policies_count']
            except:
                out = 0  # if we do have API data but it is missing this field, then just assume zero
        #print(out)
    return out


def get_contact_point(row):
    if row.is_corresponding_author_a_vu_author is True:
        res = row['corresponding_author_indexed_name_(matched)']
    else:
        res = row['first_VU_author']
    # bij een workflow moet er even op PURE gekeken worden naar de huidige faculteit/groep van de auteur (evt hand/automatisch)
    return res

def renames(df):
    df = df[[ 'eid',
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

    col_rename_dict = {'publicationName' : 'journal_name',
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
    df = df.rename(columns=col_rename_dict)

    return df


def add_abstract_columns(df_in):
    #
    # please upgrade to crystals later
    #
    df_ab = pd.DataFrame()
    for counter, cur_eid in enumerate(list(df_in.eid.unique())):
        # get abstract
        dict_ab_info = get_scopus_abstract_info(cur_eid)  # !
        dict_ab_info['eid'] = cur_eid
        df_ab = df_ab.append(dict_ab_info, ignore_index=True)

    return df_in.merge(df_ab, on='eid', how='left')


def add_author_info_columns(df_in, chosen_affid):

    df_au = pd.DataFrame()
    for counter, cur_eid in enumerate(list(df_in.eid.unique())):  # every eid = 1 paper = 1 author set = 1 routine
        # not ideal but OK
        abs_obj = df_in[df_in.eid == cur_eid].iloc[0, :].abstract_object

        # get first chosen affiliation author
        dict_auth_info = get_first_chosen_affiliation_author(abs_obj, chosen_affid)
        dict_auth_info['eid'] = cur_eid
        df_au = df_au.append(dict_auth_info, ignore_index=True)
    return df_in.merge(df_au, on='eid', how='left')


def add_faculty_info_columns(df_in, ff):

    df_ff = pd.DataFrame()
    for counter, cur_eid in enumerate(list(df_in.eid.unique())):  # every eid = 1 paper = 1 ff-return = 1 routine
        # not ideal but OK
        abs_obj = df_in[df_in.eid == cur_eid].iloc[0, :].abstract_object
        author_error = df_in[df_in.eid == cur_eid].iloc[0, :].first_affil_author_has_error
        author_org = df_in[df_in.eid == cur_eid].iloc[0, :].first_affil_author_org

        if author_error == True:
            print('no chosen affid author found at EID:' + str(cur_eid))
            dict_ff = ff.match_nan()
        else:
            # get faculty
            dict_ff = ff.match(author_org)
        dict_ff['eid'] = cur_eid
        df_ff = df_ff.append(dict_ff, ignore_index=True)

    return df_in.merge(df_ff, on='eid', how='left')
