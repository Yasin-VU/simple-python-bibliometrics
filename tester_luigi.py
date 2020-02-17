
# Goal: make a data pipeline to make the package even more user-friendly
# 
# Current status: multiple functions must be run in correct order to get data
#                 so getting data may require multiple function calls
# Target status:  just indicate which dataset you need and luigi does the rest
#                 also more robust, with caches, schedulable



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

class MyTask(luigi.Task):
    x = luigi.IntParameter()
    y = luigi.IntParameter(default=45)

    def run(self):
        print(self.x + self.y)


class MyTask1(luigi.Task):
    x = luigi.IntParameter()
    y = luigi.IntParameter(default=0) 

    def run(self):
        print(self.x + self.y)


class MyTask2(luigi.Task):
    x = luigi.IntParameter()
    y = luigi.IntParameter(default=1)
    z = luigi.IntParameter(default=2)

    def run(self):
        print(self.x * self.y * self.z)

if False:
    if __name__ == '__main__':
        luigi_run_result = luigi.build([MyTask1(x=10), MyTask2(x=15, z=3)])
        print(luigi_run_result)



class Streams(luigi.Task):
    """
    Faked version right now, just generates bogus data.
    """
    date = luigi.DateParameter()
    qq = 1
    qq = qq + 1

    def run(self):
        """
        Generates bogus data and writes 
        it into the :py:meth:`~.Streams.output` target.
        """
        with self.output().open('w') as output:
            for _ in range(1000):
                output.write('{} {} {}\n'.format(
                    random.randint(0, 999),
                    random.randint(0, 999),
                    random.randint(0, 999)))

    def output(self):
        """
        Returns the target output for this task.
        In this case, a successful execution of this task will create a file 
        in the local file system.
        :return: the target output for this task.
        :rtype: object (:py:class:`luigi.target.Target`)
        """

        return luigi.LocalTarget(PATH_START_PERSONAL + 
                                 '/luigi/data/streams_faked_%s.tsv' 
                                 % self.date)  # no date...


class Streams_scopus(luigi.Task):
    """
    Generates bogus data.
    """
    #date = luigi.DateParameter()
    yr = luigi.IntParameter()
    mn = luigi.IntParameter()

    def run(self):
        """
        Generates data and writes it into the 
        :py:meth:`~.Streams.output` target.
        """

        mini_test = True

        print('naming swapped?')
        cur_mon = self.yr
        cur_year = self.mn

        scopus_date_string = get_today_for_pubdatetxt_integers(cur_year, 
                                                               cur_mon)

        VU_with_VUMC_affid = "(   AF-ID(60001997) OR    AF-ID(60008734) OR AF-ID(60029124) OR AF-ID(60012443) OR AF-ID(60109852) OR AF-ID(60026698) OR AF-ID(60013779) OR AF-ID(60032886) OR AF-ID(60000614) OR AF-ID(60030550) OR AF-ID(60013243) OR AF-ID(60026220))"
        my_query = (VU_with_VUMC_affid 
                    + ' AND ' 
                    + "PUBDATETXT( " 
                    + scopus_date_string + " )")  
        # RECENT(1) is somehow very slow
        if mini_test:
            my_query = "TITLE(DATA) AND " + my_query


        res = pd.DataFrame(ScopusSearch(my_query, refresh=True).results)

        # luigi pandas?

        if False:
            with self.output().open('w') as output:
                for _ in range(1000):
                    output.write('{} {} {}\n'.format(
                        random.randint(0, 999),
                        random.randint(0, 999),
                        random.randint(0, 999)))
        else:
            res.to_csv(self.output().path, index=False, encoding='utf-8')

    def output(self):
        """
        Returns the target output for this task.
        In this case, a successful execution of this task will create a file in the local file system.
        :return: the target output for this task.
        :rtype: object (:py:class:`luigi.target.Target`)
        """

        return luigi.LocalTarget(PATH_START_PERSONAL + '/luigi/data/streams_faked_%s_%d.tsv' % (self.yr, self.mn))



class RefreshUnpaywall(luigi.Task):
    #date_interval = luigi.DateIntervalParameter()
    year_range = luigi.ListParameter()

    def output(self):
        return luigi.LocalTarget(PATH_START_PERSONAL + '/luigi/data/streams_faked_2.tsv') # % self.date_interval

    def requires(self):
        zip_list = [list(zip(np.arange(0, 12) + 1, [yr] * 12)) for yr in self.year_range]  # in self.year_range]
        flat_list = [item for sublist in zip_list for item in sublist]
        return [Streams_scopus(yr = year, mn = month) for year, month in flat_list]  # require multiple runs of Streams

    def run(self):

        artist_count = defaultdict(int)

        # input phase
        for input in self.input():
            res = pd.read_csv(input.path) #, index=False, encoding='utf-8')
            for cur_author_count in res.author_count.to_list():
                artist_count[cur_author_count] += 1

            #with input.open('r') as in_file:
            #    for line in in_file:
            #        _, artist, track = line.strip().split()
            #        artist_count[artist] += 1



        # output phase
        with self.output().open('w') as out_file:
            for artist, count in enumerate(artist_count):
                # print(out_file, artist, count)
                out_file.write(str(artist) + ' ' + str(count) + ' \n ')


if False:
    year_range = [2017, 2018]
    if __name__ == '__main__':
        luigi_run_result = luigi.build([ ####no need for this thanks to pipeline####Streams(date=datetime.date.today()),
                                        RefreshUnpaywall( year_range=year_range)])  # date=datetime.date.today(),
        print(luigi_run_result)



# let's do the entire oadash pipeline
# but first we need to bring all aspects to production...
# + careful with scopus-search completeness, splitting it up may cause loss of data


# a = input('press any key')
print('continuing')
# oadash-ETL buildup

# settings
#
# most-outer-loop-settings: variable
year_range_outer = [2000, 2001, 2002]
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
         'citedby_count', 'fund_sponsor', 'aggregationType', 'openaccess']
        df = df[fav_fields]  # cut fields
        #
        # 1X: drop all empty eids to prevent issues later (to be safe)
        df = df.dropna(axis=0, subset=['eid'], inplace=False)

        print(len(df))
        df.to_pickle(self.output().path) #, encoding='utf-8')

    def output(self):
        """
        Returns the target output for this task.
        In this case, a successful execution of this task will create a file in the local file system.
        :return: the target output for this task.
        :rtype: object (:py:class:`luigi.target.Target`)
        """

        return luigi.LocalTarget(PATH_START_PERSONAL + '/luigi/data/scopus_years_%s_%s.pkl' % (self.yr, self.qr))


class AddYearAndMonth(luigi.Task):
    """
    adds year and month columns to scopus raw data
    """
    yr = luigi.IntParameter()
    qr = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(PATH_START_PERSONAL + '/luigi/data/scopus_years_dated_%s_%s.pkl' % (self.yr, self.qr))

    def requires(self):
        return ScopusPerYear(yr=self.yr, qr=self.qr)

    def run(self):

        # input and processing phase
        for input in self.input():  # should be just 1 for this routine
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
    processing_args = luigi.Parameter()  # hash

    def output(self):
        return luigi.LocalTarget(PATH_START_PERSONAL
                                  + '/luigi/data/'
                                  + self.out_path_name_prefix
                                  + '_%s_%s.pkl' % (self.yr, self.qr))

    def requires(self):
        req_fn = pickle.loads(self.required_luigi_class)
        return req_fn(yr=self.yr, qr=self.qr)

    def run(self):

        # input phase
        df_out = pd.read_pickle(self.input().path) #, index=False, encoding='utf-8')


        # you are here: code seems to be working but we need a check cause I got 5/5 author errors... yuples were fixed

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
        print(df_out.head(1).T)

        ###print(df_out[['first_affil_author', 'first_affil_author_has_error', 'first_affil_author_org']])

        # output phase
        df_out.to_pickle(self.output().path) #, encoding='utf-8')


# fill instance inherent args in, leave rest open for variable runs
AddAbstractColumns = partial(AddX,
                             # yr=2020,
                             # qr='TITLE(TENSOR data)',
                             out_path_name_prefix='scopus_years_abs',
                             required_luigi_class=pickle.dumps(ScopusPerYear),
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
                         processing_args=pickle.dumps([False])
                         )

#### do other decos work? add_unpaywall_columns add_altmetric_columns

"""
# we need a wrapper for this to bring it to the same form as add_author_info_columns
# afterwards move it to core_functions
def add_deal_info_columns():
    # wrap add_deal_info()
    pass

AddDealColumns = partial(AddX,
                         out_path_name_prefix='scopus_years_deals',
                         required_luigi_class=pickle.dumps(AddUnpaywallColumns),
                         processing_function=pickle.dumps(add_deal_info_columns),
                         processing_args=[path_deals, path_isn]
                         )
"""

# afterwards steps 8, 9, 10, 11, 12...
                         
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



# PS: the database functions need to be generalized s.t. ppl can plug own server
#     or use flat-files instead [preferred]
  


######################################

qq=1
qq+=1



year_range = [2017, 2018]
if __name__ == '__main__':
    #luigi_run_result = luigi.build([ScopusPerYear(yr=2020, qr='TITLE(TENSOR data)')])  # date=datetime.date.today(),

    #luigi_run_result = luigi.build([AddAbstractColumns(yr=2020, qr='TITLE(TENSOR data)')])

    #luigi_run_result = luigi.build([AddAuthorInfoColumns(yr=2020, qr=' AF-ID(60008734) AND TITLE(DATA) ')])

    luigi_run_result = luigi.build([AddUnpaywallColumns(yr=2020, qr=' AF-ID(60008734) AND TITLE(DATA) ')])
    print(luigi_run_result)





# luigi design
# we have a few functions now, let's try to push some real data through?


















a = input('press any key to run the regular pipeline')

### below is the starting point pipeline

# major tasks:
# 1x. perform scopus search
s = ScopusSearch(my_query, refresh=True)  #(VU_aff + " AND " + recent, refresh=True)
df = pd.DataFrame(s.results)
fav_fields = ['eid',  'creator',  'doi',  'title',  'afid',
 'affilname',  'author_count',  'author_names',  'author_afids',
 'coverDate',  'coverDisplayDate',  'publicationName', 'issn',  'source_id', 'eIssn',
 'citedby_count', 'fund_sponsor', 'aggregationType', 'openaccess']
df = df[fav_fields]  # cut fields
#
# 1X: drop all empty eids to prevent issues later (to be safe)
df = df.dropna(axis=0, subset=['eid'], inplace=False)

# 2. add year and month
df = add_year_and_month(df)  # add info columns

# below we split up 3 routines which are wrongfully combined (refactor required)

# 3. add abs
df = add_abstract_columns(df)

# 4. add au
df = add_author_info_columns(df, chosen_affid)

# 5. add ff
#
# 5A. prepare the faculty_finder NLP tool
org_info = pd.read_excel(path_org, skiprows=0)
ff = faculty_finder(organizational_chart=org_info)
#
# 5B. add ff
df = add_faculty_info_columns(df, ff)

# 6. add unpaywall
df = add_unpaywall_columns(df, silent=False)

# 7. add deal info
df = add_deal_info(path_deals=path_deals, path_isn=path_isn, df_b=df)

# 8. add corresponding author info
df = (corresponding_author_functions()
      .add_corresponding_author_info(df=df,
                                     vu_afids=vu_afids,
                                     ukb_afids=all_vsnu_sdg_afids))

# 9. post-process unpaywall
df['upw_oa_color_category'] = df.upw_oa_color.apply(fn_cats)
df['upw_oa_color_verbose'] = df['upw_oa_color'].apply(lambda x: 'unknown' if x is np.nan else x)

# 10. renames
df = renames(df)

# 11. get contact person
df['vu_contact_person'] = df.apply(get_contact_point,axis=1)

# 12. after this point, there is a lot going on with PURE processing, P+S merging, keuzemodel, STM(tester_STM!!),
#     and more stuff (partially in production, partially not yet)
#     this will require a lot of searching and refactoring
#     we must first move the stuff above to luigi s.t. we can skip those edits/downloads while testing : )

df.head()




















