import numpy as np
from static import PATH_START, PATH_START_PERSONAL
from static import PATH_START_SERVER , PATH_START_PERSONAL_SERVER
import pandas as pd
from core_functions import crystal_scopus_abstract, crystal_scopus_abstract2
import time


# ! VALIDATION IS REQUIRED


MTCOUNT = 10  #00
df = pd.read_excel(PATH_START  + r'raw data algemeen\api_caches\try02.xlsx', nrows=MTCOUNT, usecols=['eid'])
print(df.head())

#res = crystal_scopus_abstract2(df.head(2))



t0 = time.time()
fullres = crystal_scopus_abstract2(df.head(MTCOUNT), multi_thread=True)
t1 = time.time()
print(t1-t0)
print('we just did ' + str(MTCOUNT) + ' records in just ' + str(t1-t0) + ' seconds!')

print(fullres.scopus_abstract_text.isnull().mean())
print(fullres.scopus_abstract_text.isnull().mean())
print(fullres.scopus_abstract_retries.mean())
print(fullres.scopus_abstract_retries.max())

# ST
#we just did 100 records in just 124.84675812721252 seconds!
#0.04
#0.04
#0.0
#0.0


qq=1
qq+=1


#input('nu productie?')

# go for it

start_path = 'E:/Shared drives/Aurora-SDG-analysis/Aurora-SDG-Analysis-project02/02-query-crafting/SDG-Survey/sdg-survey-result-data/'
df_eids = pd.read_csv(start_path + 'eids.csv')

#df_eids = df_eids.head(102)

bits=10
stepsize = int(np.ceil(len(df_eids) / bits)+1)

for cur_bit in np.arange(0,bits):
    print('-------')
    print(cur_bit)

    df_eids_CUR = df_eids.iloc[stepsize*cur_bit: stepsize*(cur_bit+1),:]

    if len(df_eids_CUR) > 0:
        t0 = time.time()
        fullres = crystal_scopus_abstract2(df_eids_CUR, multi_thread=True)
        t1 = time.time()
        print(t1-t0)
        print('we just did ' + str(len(df_eids_CUR)) + ' records in just ' + str(t1-t0) + ' seconds!')

        print(fullres.scopus_abstract_text.isnull().mean())
        print(fullres.scopus_abstract_text.isnull().mean())
        print(fullres.scopus_abstract_retries.mean())
        print(fullres.scopus_abstract_retries.max())

        fullres[['eid', 'scopus_abstract_text']].to_csv(start_path + 'experimental_abstract_texts' + str(cur_bit) + '.csv')

# we validated it now I guess
