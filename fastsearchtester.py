from pybliometrics.scopus import ScopusSearch
import numpy as np
from static import PATH_START, PATH_START_PERSONAL
from static import PATH_START_SERVER , PATH_START_PERSONAL_SERVER
import pandas as pd
from core_functions import crystal_scopus_abstract, crystal_scopus_abstract2
import time


start_path = 'E:/Shared drives/Aurora-SDG-analysis/Aurora-SDG-Analysis-project02/02-query-crafting/SDG-Survey/sdg-survey-result-data/'
df_eids = pd.read_csv(start_path + 'eids.csv')

eid_list = df_eids.eid.to_list()

phase_one = True

if phase_one:
    ii = 0
    while len('EID( ' + ' ) OR EID( '.join(eid_list[100*ii:100*(ii+1)]) + ' )') > 7+3:
        print(ii)
        qr = 'EID( ' + ' ) OR EID( '.join(eid_list[100*ii:100*(ii+1)]) + ' )'
        s = ScopusSearch(qr)
        df = pd.DataFrame(s.results)
        df.to_csv(start_path + '/yasin_test_zone/' + str(ii) + '.csv')
        ii = ii + 1
        time.sleep(0.1)


else:
    df_total = pd.DataFrame()
    for jj in np.arange(0, 106+1):
        df_part = pd.read_csv(start_path + '/yasin_test_zone/' + str(jj) + '.csv')
        df_total = df_total.append(df_part)
    print(len(df_total))
    df_total = df_total.drop_duplicates(subset='eid')  # no dupe eids, we are going to merge on eid
    print(len(df_total))
    df_total.to_csv(start_path + '/yasin_test_zone/' + 'total_scopus_search_of_10683_eids.csv')


print(len(df_total))
print(len(df_eids))

# some eids in df_eids are void like 'nnnn', some are nan and some are invalid and thus no scopussearch return for them

q = 1
q = q + 1
