


# imports
#
from static import PATH_START, PATH_START_PERSONAL
from static import PATH_START_SERVER , PATH_START_PERSONAL_SERVER
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, PATH_START_PERSONAL + '/common_functions')  # not needed sometimes
from nlp_functions import stack_titles, SoftTitleMatcher
import matplotlib.pyplot as plt
#
# end of imports





core = 'oa2019map'

# settings
#
# starting point of P+S table to improve with STM
df_total = pd.read_csv(PATH_START + '/raw data algemeen/' + core + '/merged_data/df_total.csv')
# df_total has all multi_years
# chosen year
chosen_year = 2019
#
# end of settings


# test
#
stm = SoftTitleMatcher()

# df_total: 18k
# both      6304
# pure      4104
# scopus    7593
#
# df_total_with_STM_rich_2018.groupby('merge_source').agg('count').max(1)


df_total_with_STM, df_total_with_STM_rich_2018 = stm.improve_merged_table_using_STM_results(df_total=df_total,
                                                                                            chosen_year=chosen_year,
                                                                                            out_path=None,
                                                                                            do_save=False,
                                                                                            cond_len=4,
                                                                                            cond_score=0.6)

print('done')
print(PATH_START + '/raw data algemeen/' + core + '/merged_data/refactor_test.xlsx')

print('----')
print(len(df_total))
print(df_total.groupby('merge_source').agg('count').max(1))
print(df_total[(df_total.scopus_year == 2019) | (df_total.pure_year == 2019)].groupby('merge_source').agg('count').max(1))
print(df_total.groupby(['year', 'merge_source']).agg('count').max(1))
print('----')
print(len(df_total_with_STM_rich_2018))
print(df_total_with_STM_rich_2018.groupby('merge_source').agg('count').max(1))
print(df_total_with_STM_rich_2018[(df_total_with_STM_rich_2018.scopus_year == 2019)
                                  | (df_total_with_STM_rich_2018.pure_year == 2019)]
      .groupby('merge_source')
      .agg('count')
      .max(1))
print(df_total_with_STM_rich_2018.groupby(['year', 'merge_source']).agg('count').max(1))
print('----')


qq=1
qq=qq+1

df_total_with_STM_rich_2018.to_csv(PATH_START + '/raw data algemeen/' + core + '/merged_data/refactor_test.csv')
example_data = pd.read_excel(PATH_START +
                           r'raw data algemeen\code speedup test data\nlp2_result_fast - refactor test.xlsx')  # fixed

#
# end of test
#
# ISSUES:
# 1. there are issues with packages and some pandas warnings: tackle them please
# 2. we need a multi-year approach for STM too: how to implement that?: df_unmerged_P/S have all 3 years: use that
#    and then we need to post-filter within the Power BI or right before that and only mail the middle year...
#



