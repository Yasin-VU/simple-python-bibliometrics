
# !!! TESTING INCOMPLETE !!!


# imports
#
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, 'C:/Users/yasin/Desktop/git/common_functions')  # not needed sometimes
from nlp_functions import stack_titles, SoftTitleMatcher
import matplotlib.pyplot as plt
#
# end of imports


# settings
#
# starting point of P+S table to improve with STM
df_total = pd.read_csv(r'G:\UBVU\Data_RI\raw data algemeen\pipeline_test\merged_data\df_total.csv')
# chosen year
chosen_year = 2018
#
# end of settings


# test
#
stm = SoftTitleMatcher()


df_total_with_STM, df_total_with_STM_rich_2018 = stm.improve_merged_table_using_STM_results(df_total=df_total,
                                                                                            chosen_year=chosen_year,
                                                                                            out_path=None,
                                                                                            do_save=False,
                                                                                            cond_len=4,
                                                                                            cond_score=0.6)
df_total_with_STM_rich_2018.to_csv(r'G:\UBVU\Data_RI\raw data algemeen\code speedup test data\refactor_test.xlsx')
tester = pd.read_csv(r'G:\UBVU\Data_RI\raw data algemeen\code speedup test data\nlp2_result_fast - refactor test.xlsx')
#
# end of test
#
# there are issues with packages and some pandas warnings: tackle them please




