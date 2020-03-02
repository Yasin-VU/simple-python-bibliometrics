
# Goal: make a data pipeline to make the package even more user-friendly
# 
# Current status: multiple functions must be run in correct order to get data
#                 so getting data may require multiple function calls
# Target status:  just indicate which dataset you need and luigi does the rest
#                 also more robust, with caches, schedulable
#
# update: scopus arm now runs, but settings disallow > 20k records, so will re-test with new settings now in .cfg ...

# this single line will allow luigi to run longer tasks :)
LUIGI_CONFIG_PATH = 'C:/Users/yasin/Desktop/git/simple-python-bibliometrics/example.cfg'

# my_module.py, available in your sys.path
import luigi
import random
from static import PATH_START, PATH_START_PERSONAL
from static import PATH_START_SERVER , PATH_START_PERSONAL_SERVER
from collections import defaultdict
import datetime
from pybliometrics.scopus import ScopusSearch
import pandas as pd
import numpy as np
from core_functions import get_today_for_pubdatetxt
from core_functions import get_today_for_pubdatetxt_super
from core_functions import get_today_for_pubdatetxt_integers
from core_functions import get_contact_point
from core_functions import add_year_and_month
from core_functions import get_scopus_abstract_info
from core_functions import get_first_chosen_affiliation_author  
from core_functions import add_unpaywall_columns, add_altmetric_columns
from core_functions import my_timestamp
from core_functions import add_deal_info
from core_functions import add_abstract_columns
from core_functions import add_author_info_columns
from core_functions import add_faculty_info_columns
from core_functions import fn_cats
from core_functions import renames
from nlp_functions import faculty_finder
from nlp_functions import corresponding_author_functions
import pickle
from functools import partial
import time  # imports at the top

# settings
#
# most-outer-loop-settings: variable
path_out = PATH_START_PERSONAL + '/oa new csv/'  # no r
chosen_affid = ["60008734","60029124","60012443","60109852","60026698","60013779","60032886","60000614",
                "60030550","60013243","60026220","60001997"]  # I added 60001997 and thus I added VUMC
#VU_noMC_affid = "(AF-ID(60008734) OR AF-ID(60029124) OR AF-ID(60012443) OR AF-ID(60109852) OR AF-ID(60026698) OR AF-ID(60013779) OR AF-ID(60032886) OR AF-ID(60000614) OR AF-ID(60030550) OR AF-ID(60013243) OR AF-ID(60026220))"
VU_with_VUMC_affid = "(   AF-ID(60001997) OR    AF-ID(60008734) OR AF-ID(60029124) OR AF-ID(60012443) OR AF-ID(60109852) OR AF-ID(60026698) OR AF-ID(60013779) OR AF-ID(60032886) OR AF-ID(60000614) OR AF-ID(60030550) OR AF-ID(60013243) OR AF-ID(60026220))"
my_query = VU_with_VUMC_affid + ' AND ' + "( PUBYEAR  =  2018) " + "TITLE(TENSOR)"  # "PUBDATETXT(February 2018)"
#
#
#
# most-outer-loop-settings: semi-fixed
path_deals = PATH_START + '/raw data algemeen/apcdeals.csv'
path_isn = PATH_START + '/raw data algemeen/ISN_ISSN.csv'
path_org = PATH_START + '/raw data algemeen/vu_organogram_2.xlsx'
path_vsnu_afids = PATH_START + '/raw data algemeen/afids_vsnu_nonfin.csv'
vu_afids = chosen_affid
# this is vsnu w/o phtu and such (borrowed from VSNU-SDG-data), but should approach the UKB list... good for now. update later.
all_vsnu_sdg_afids = pd.read_csv(path_vsnu_afids).iloc[:,1].astype('str').to_list()
#
# end of settings


# now step by step push into an ETL-form in order to be able to easily skip steps during testing


def my_hash(cur_string):
    # deterministic and poor hash but at least reproducible
    return str(len(cur_string)) + cur_string[0:3] + cur_string[-3:]


class ScopusPerYear(luigi.Task):
    """
    Harvests one year of Scopus for a given query
    """
    yr = luigi.IntParameter()
    qr = luigi.Parameter()

    def run(self):
        """
        Generates data and writes it into the :py:meth:`~.Streams.output` target.
        """

        cur_year = self.yr
        cur_query = self.qr

        run_query = cur_query + ' AND ( PUBYEAR  =  ' + str(cur_year) + ') '

        size = ScopusSearch(run_query, refresh=True, download=False).get_results_size()

        if size > 10000:
            print('scopus query with over 10k records running, careful')

        df = pd.DataFrame(ScopusSearch(run_query, refresh=True).results)

        fav_fields = ['eid',  'creator',  'doi',  'title',  'afid',
         'affilname',  'author_count',  'author_names',  'author_afids',
         'coverDate',  'coverDisplayDate',  'publicationName', 'issn',  'source_id', 'eIssn',
         'citedby_count', 'fund_sponsor', 'aggregationType', 'openaccess', 'description', 'authkeywords']
        df = df[fav_fields]  # cut fields
        #
        # 1X: drop all empty eids to prevent issues later (to be safe)
        df = df.dropna(axis=0, subset=['eid'], inplace=False)

        #print(len(df))
        df.to_pickle(self.output().path) #, encoding='utf-8')

    def output(self):
        """
        Returns the target output for this task.
        In this case, a successful execution of this task will create a file in the local file system.
        :return: the target output for this task.
        :rtype: object (:py:class:`luigi.target.Target`)
        """

        return luigi.LocalTarget(PATH_START_PERSONAL + '/luigi/data/scopus_years_%s_%s.pkl' % (self.yr, my_hash(self.qr)))


class AddYearAndMonth(luigi.Task):
    """
    adds year and month columns to scopus raw data
    """
    yr = luigi.IntParameter()
    qr = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(PATH_START_PERSONAL + '/luigi/data/scopus_years_dated_%s_%s.pkl' % (self.yr, my_hash(self.qr)))

    def requires(self):
        return ScopusPerYear(yr=self.yr, qr=self.qr)

    def run(self):

        # input and processing phase
        input = self.input()  # should be just 1 for this routine
        df = pd.read_pickle(input.path) #, index=False, encoding='utf-8')
        df = add_year_and_month(df, 'coverDate')  # add info columns

        # output phase
        df.to_pickle(self.output().path) #, encoding='utf-8')

######################################


class AddX(luigi.Task):
    """
    adds columns to data based on chosen settings, and set io
    """
    yr = luigi.IntParameter()
    qr = luigi.Parameter()
    out_path_name_prefix = luigi.Parameter()
    required_luigi_class = luigi.Parameter()
    processing_function = luigi.Parameter()
    processing_args = luigi.Parameter()  # my_hash

    def output(self):
        return luigi.LocalTarget(PATH_START_PERSONAL
                                  + '/luigi/data/'
                                  + self.out_path_name_prefix
                                  + '_%s_%s.pkl' % (self.yr, my_hash(self.qr)))

    def requires(self):
        req_fn = pickle.loads(self.required_luigi_class)
        return req_fn(yr=self.yr, qr=self.qr)

    def run(self):

        # input phase
        df_out = pd.read_pickle(self.input().path) #, index=False, encoding='utf-8')

        # processing phase
        #
        proc_fn = pickle.loads(self.processing_function)
        #

        proc_args = pickle.loads(self.processing_args) #self.processing_args  #

        #print(self.processing_args)
        if len(proc_args) > 0:
            arg_fn = []
            for element in proc_args:
                #print(element)
                if (type(element) is tuple):
                    element = list(element)
                arg_fn.append(element)
            df_out = proc_fn(df_out, *arg_fn)
        else:
            df_out = proc_fn(df_out)

        # debug
        #print(df_out.head(1).T)
        #print(df_out.iloc[:, -1])

        # output phase
        df_out.to_pickle(self.output().path) #, encoding='utf-8')


# fill instance inherent args in, leave rest open for variable runs
AddAbstractColumns = partial(AddX,
                             # yr=2020,
                             # qr='TITLE(TENSOR data)',
                             out_path_name_prefix='scopus_years_abs',
                             required_luigi_class=pickle.dumps(AddYearAndMonth),
                             processing_function=pickle.dumps(add_abstract_columns),
                             processing_args=pickle.dumps([])
                             )


AddAuthorInfoColumns = partial(AddX,
                             # yr=2020,
                             # qr='TITLE(TENSOR data)',
                             out_path_name_prefix='scopus_years_au',
                             required_luigi_class=pickle.dumps(AddAbstractColumns),
                             processing_function=pickle.dumps(add_author_info_columns),
                             processing_args=pickle.dumps([[*chosen_affid + ['0']]])  # quick-fix for luigi tuple issue
                             )


org_info = pd.read_excel(path_org, skiprows=0)
ff = faculty_finder(organizational_chart=org_info)
AddFFColumns = partial(AddX,
                         out_path_name_prefix='scopus_years_ff',
                         required_luigi_class=pickle.dumps(AddAuthorInfoColumns),
                         processing_function=pickle.dumps(add_faculty_info_columns),
                         processing_args=pickle.dumps([ff])
                         )


# untested, just prepped: see below


AddUnpaywallColumns = partial(AddX,
                         out_path_name_prefix='scopus_years_upw',
                         required_luigi_class=pickle.dumps(AddFFColumns),
                         processing_function=pickle.dumps(add_unpaywall_columns),
                         processing_args=pickle.dumps([])  # removed False
                         )

#### do other decos work? add_unpaywall_columns add_altmetric_columns


# YOU ARE HERE: ISSUE: PARAMETER ORDERING MIXUP
#
# we need a wrapper for this to bring it to the same form as add_author_info_columns
# afterwards move it to core_functions
def add_deal_info_reorder(df_b_in, path_deals_in, path_isn_in):
    # reorder args
    return add_deal_info(path_deals_in, path_isn_in, df_b_in)


add_deal_info_columns = partial(add_deal_info_reorder,
                                path_deals_in=path_deals,
                                path_isn_in=path_isn)


AddDealColumns = partial(AddX,
                         out_path_name_prefix='scopus_years_deals',
                         required_luigi_class=pickle.dumps(AddUnpaywallColumns),
                         processing_function=pickle.dumps(add_deal_info_columns),
                         processing_args=pickle.dumps([])
                         )


add_corr_aut_columns = partial(corresponding_author_functions().add_corresponding_author_info,
                               vu_afids=vu_afids,
                               ukb_afids=all_vsnu_sdg_afids)

AddCorrAutColumns = partial(AddX,
                            out_path_name_prefix='scopus_years_coraut',
                            required_luigi_class=pickle.dumps(AddDealColumns),
                            processing_function=pickle.dumps(add_corr_aut_columns),
                            processing_args=pickle.dumps([])
                            )


def add_extra_unpaywall_columns(df_in):
    """
    adds extra unpaywall columns to a dataframe with unpaywall info
    :param df: the dataframe to start with, it must have a column upw_oa_color with unpaywall colors
    which fit the keys of fn_cats() and not have the column names upw_oa_color_category and upw_oa_color_verbose
    :return: the same df with 2 extra columns named upw_oa_color_category and upw_oa_color_verbose,
    which are resp. an encoded version and a version where missing values are replaced by text as 'unknown'
    where you have to be careful with running null checks on as nulls have become plain text
    """
    df_in['upw_oa_color_category'] = df_in.upw_oa_color.apply(fn_cats)
    df_in['upw_oa_color_verbose'] = df_in['upw_oa_color'].apply(lambda x: 'unknown' if x is np.nan else x)

    return df_in

AddXUnpaywallColumns = partial(AddX,
                            out_path_name_prefix='scopus_years_upwx',
                            required_luigi_class=pickle.dumps(AddCorrAutColumns),
                            processing_function=pickle.dumps(add_extra_unpaywall_columns),
                            processing_args=pickle.dumps([])
                            )

# ! careful ! the renames function works here because the functionality is through sheer luck compatible
# with the functionality of 'df=add_stuff(df)', but this will generally not be the case
# that is, if you make one with a different dataframe manipulation it may fail somewhere in the chain of functions
# also, it might also DIRECTLY fail when we update AddX, since 'renames' does not fit in its goal-scope
# even though it 'works' right now
# take home message: please only use AddX for add_X_columns types of functions, and edit this out some day (marked '!')
Renames = partial(AddX,
                  out_path_name_prefix='scopus_years_renamed',
                  required_luigi_class=pickle.dumps(AddXUnpaywallColumns),
                  processing_function=pickle.dumps(renames),
                  processing_args=pickle.dumps([])
                  )


def add_contact_person_columns(df_in):
    df_in['vu_contact_person'] = df_in.apply(get_contact_point, axis=1)
    return df_in

AddContactPersonColumns = partial(AddX,
                                  out_path_name_prefix='scopus_years_complete',
                                  required_luigi_class=pickle.dumps(Renames),
                                  processing_function=pickle.dumps(add_contact_person_columns),
                                  processing_args=pickle.dumps([])
                                  )


AddAltmetricColumns = partial(AddX,
                              out_path_name_prefix='scopus_years_altm',
                              required_luigi_class=pickle.dumps(AddContactPersonColumns),
                              processing_function=pickle.dumps(add_altmetric_columns),
                              processing_args=pickle.dumps([])
                              )




class MultiScopusEndPoint(luigi.Task):
    year_range = luigi.ListParameter()
    qr = luigi.Parameter()
    # AddContactPersonColumns

    def output(self):
        return luigi.LocalTarget(PATH_START_PERSONAL
                                  + '/luigi/data/'
                                  + 'scopus_multi'
                                  + '_%s_%s.pkl' % (str(self.year_range), my_hash(self.qr)))

    def requires(self):
        return [AddAltmetricColumns(yr=year, qr=self.qr) for year in self.year_range]

    def run(self):


        # input phase
        df_out = pd.DataFrame()
        for cur_input in self.input():
            res = pd.read_pickle(cur_input.path) #, index=False, encoding='utf-8')
            df_out = df_out.append(res)

        # output phase
        df_out.to_pickle(self.output().path) #, encoding='utf-8')


class MultiScopusEndPoint_csv(luigi.Task):
    year_range = luigi.ListParameter()
    qr = luigi.Parameter()
    # AddContactPersonColumns

    def output(self):
        return luigi.LocalTarget(PATH_START_PERSONAL
                                  + '/luigi/data/'
                                  + 'scopus_multi_csv'
                                  + '_%s_%s.csv' % (str(self.year_range), my_hash(self.qr)))

    def requires(self):
        return MultiScopusEndPoint(year_range=self.year_range, qr=self.qr)

    def run(self):

        # input phase
        df_out = pd.read_pickle(self.input().path) #, index=False, encoding='utf-8')
        #df_out = df_out.drop(columns=['abstract_object'])  # most basic way but OK

        # output phase
        df_out.to_csv(self.output().path) #, encoding='utf-8')


# now to db (!)


# the steps after 12 need to be plotted
# so what does happen next?:
#
# A. the PURE integration
# A1. pure read in and preprocess
# A2. pure scopus-steps replication including skipping unpaywall [refactor!]
# A3. 3-method merger scopus and pure
# A4. STM postmerge merger
# A5. columndistiller
# that ends the data for theoretic product 1
# but there are more routes
# there is the ML route as well, that one is also useful (think faculty finder)
# there is also the topic_analysis route, that one is clean and perhaps shareable?
#
# B. ?
#
#
#
# The major issue here is that the old single-function-does-all mentality will not work here
# We actually want to have well-defined well-split pipe parts! A single pipe piece is not a pipeline
# and would miss out on features like intermediate checkpointing and parallel runs per year
# and also indirectly saving results per year for re-use and interval-sharing across projects
#
# update: we cannot continue until we finish the refactoring test of tester_soft_title_matcher.py (!)
# I checked the refactored code and there are bugs... they are inside the STM class during write operations
# but I suspect there are empty columns as well
# this is going to take some time, because not only do we need to push it in luigi-format, but also we need to refactor
# first s.t. the code is maintainable and less hard to split into pipes
# maybe I will make a fresh refactor with a focus on luigi-compatibility... (remember to use "nonfork version of nlp4")
#
# update2: there is much more refactoring work than I thought: there are zero top-level functions still
#          maybe we should first wrap the previous refactor and refactor that, I blocked a few moments for this

# PS: the database functions need to be generalized s.t. ppl can plug own server
#     or use flat-files instead [preferred]


if __name__ == '__main__':
    print('starting')
    start = time.time()
    print(start)
    print(VU_with_VUMC_affid)
    mini_test = True
    if mini_test:
        task_at_hand = [MultiScopusEndPoint_csv(year_range=[2018, 2019, 2020], qr=' AF-ID(60008734) AND TITLE(DATA) ')]
    else:
        # needs a rerun(!), but won't overwrite the other hash (almost always)
        task_at_hand = [MultiScopusEndPoint(year_range=[2009, 2010, 2011, 2012, 2013,
                                                        2014, 2015, 2016, 2017, 2018,
                                                        2019, 2020],
                                            qr=' ' + VU_with_VUMC_affid + ' ')]

    luigi_run_result = luigi.build(task_at_hand)
    print(luigi_run_result)
    end = time.time()
    print(end-start)
    print('done')




