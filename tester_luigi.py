
# this is a work in progress to organize all different functions into an ETL pipeline in order to provide an overview
# basically it does exactly the same, but is easier to overview, to fix, and allows for ETL-functions*
# * like skipping where data is already available and scheduling and advanced error logs, etc


# my_module.py, available in your sys.path
import luigi
import random
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
from core_functions import get_first_chosen_affiliation_author  # ! i want all vu authors now : )
from core_functions import add_unpaywall_columns
from core_functions import my_timestamp
from core_functions import add_deal_info
from core_functions import add_abstract_columns, add_author_info_columns, add_faculty_info_columns, fn_cats, renames
from nlp_functions import faculty_finder
from nlp_functions import corresponding_author_functions


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
        luigi_run_result = luigi.build([MyTask1(x=10), MyTask2(x=15, z=3)])  # , detailed_summary=True
        print(luigi_run_result)

# now I want an unpaywall task : )
# PS: notice that you need cron/windows scheduler to run this orchestrator py file
#     however, that can be set up easily, and right now I first want to tackle the biggest issue
#     which is not pressing 'go' manually once in a while, but rather the pipeline being in multiple places


class Streams(luigi.Task):
    """
    Faked version right now, just generates bogus data.
    """
    date = luigi.DateParameter()
    qq = 1
    qq = qq + 1

    def run(self):
        """
        Generates bogus data and writes it into the :py:meth:`~.Streams.output` target.
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
        In this case, a successful execution of this task will create a file in the local file system.
        :return: the target output for this task.
        :rtype: object (:py:class:`luigi.target.Target`)
        """

        return luigi.LocalTarget('C:/Users/yasin/Desktop/luigi/data/streams_faked_%s.tsv' % self.date)  # no date...


class Streams_scopus(luigi.Task):
    """
    Generates bogus data.
    """
    #date = luigi.DateParameter()
    yr = luigi.IntParameter()
    mn = luigi.IntParameter()

    def run(self):
        """
        Generates data and writes it into the :py:meth:`~.Streams.output` target.
        """

        mini_test = True

        cur_mon = self.yr
        cur_year = self.mn

        scopus_date_string = get_today_for_pubdatetxt_integers(cur_year, cur_mon)

        VU_with_VUMC_affid = "(   AF-ID(60001997) OR    AF-ID(60008734) OR AF-ID(60029124) OR AF-ID(60012443) OR AF-ID(60109852) OR AF-ID(60026698) OR AF-ID(60013779) OR AF-ID(60032886) OR AF-ID(60000614) OR AF-ID(60030550) OR AF-ID(60013243) OR AF-ID(60026220))"
        my_query = VU_with_VUMC_affid + ' AND ' + "PUBDATETXT( " + scopus_date_string + " )"  # RECENT(1) is somehow very slow
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

        return luigi.LocalTarget('C:/Users/yasin/Desktop/luigi/data/streams_faked_%s_%d.tsv' % (self.yr, self.mn))



class RefreshUnpaywall(luigi.Task):
    #date_interval = luigi.DateIntervalParameter()
    year_range = luigi.ListParameter()

    def output(self):
        return luigi.LocalTarget('C:/Users/yasin/Desktop/luigi/data/streams_faked_2.tsv') # % self.date_interval

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
#
#
path_deals = r'G:\UBVU\Data_RI\raw data algemeen\apcdeals.csv'
path_isn = r'G:\UBVU\Data_RI\raw data algemeen\ISN_ISSN.csv'
path_org = r'G:\UBVU\Data_RI\raw data algemeen\vu_organogram_2.xlsx'
path_out = 'C:/Users/yasin/Desktop/oa new csv/'  # no r
path_vsnu_afids = r'G:\UBVU\Data_RI\raw data algemeen\afids_vsnu_nonfin.csv'
chosen_affid = ["60008734","60029124","60012443","60109852","60026698","60013779","60032886","60000614",
                "60030550","60013243","60026220","60001997"]  # I added 60001997 and thus I added VUMC
#VU_noMC_affid = "(AF-ID(60008734) OR AF-ID(60029124) OR AF-ID(60012443) OR AF-ID(60109852) OR AF-ID(60026698) OR AF-ID(60013779) OR AF-ID(60032886) OR AF-ID(60000614) OR AF-ID(60030550) OR AF-ID(60013243) OR AF-ID(60026220))"
VU_with_VUMC_affid = "(   AF-ID(60001997) OR    AF-ID(60008734) OR AF-ID(60029124) OR AF-ID(60012443) OR AF-ID(60109852) OR AF-ID(60026698) OR AF-ID(60013779) OR AF-ID(60032886) OR AF-ID(60000614) OR AF-ID(60030550) OR AF-ID(60013243) OR AF-ID(60026220))"
my_query = VU_with_VUMC_affid + ' AND ' + "( PUBYEAR  =  2018)"  + "TITLE(TENSOR)" ### "PUBDATETXT(February 2018)"

# corresponding author
vu_afids = chosen_affid
# this is vsnu w/o phtu and such (borrowed from VSNU-SDG-data), but should approach the UKB list... good for now. update later.
all_vsnu_sdg_afids = pd.read_csv(path_vsnu_afids).iloc[:,1].astype('str').to_list()


# major tasks:
# 1. perform scopus search
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
df = add_year_and_month(df, 'coverDate')  # add info columns

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
#     and all the stuff that is already in production
#     this will require a lot of searching and refactoring
#     we must first move the stuff above to luigi s.t. we can skip those edits/downloads while testing : )

df.head()




















