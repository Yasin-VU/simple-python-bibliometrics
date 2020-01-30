import numpy as np
import pandas as pd
from core_functions import crystal_scopus_abstract
import time


# ! VALIDATION IS REQUIRED


MTCOUNT = 1000
df = pd.read_excel(r'G:\UBVU\Data_RI\raw data algemeen\api_caches\try02.xlsx', nrows=MTCOUNT, usecols=['eid'])
print(df.head())

res = crystal_scopus_abstract(df.head(1))
time.sleep(1)
# check a result
import pickle
print('------')
print(pickle.loads((res.scopus_abstract_obje[0])).authorgroup)
# it works :)

# I am not sure if it will have issues due to scopus throwing errors while multi-threading
# so you must check that thoroughly!
# !

print('start')

t0 = time.time()
fullres = crystal_scopus_abstract(df.head(MTCOUNT))
t1 = time.time()
print(t1-t0)
print('we just did ' + str(1000) + ' records in just ' + str(t1-t0) + ' seconds!')

print(fullres.scopus_abstract_obje.isnull().mean())
print(fullres.scopus_abstract_obje.dropna().apply(lambda x: pickle.loads(x).title).isnull().mean())  # should be 0, a check!

print(fullres.scopus_abstract_retries.mean())
print(fullres.scopus_abstract_retries.max())


qq=1
qq=qq+1

print('warp')

if False:
    lst = []
    from pybliometrics.scopus import AbstractRetrieval
    t0 = time.time()
    for ii in np.arange(0,10):
        cur_eid = df.loc[ii,'eid']
        minires = AbstractRetrieval(identifier=cur_eid, view='FULL', refresh=True, id_type='eid')
        try:
            qq = minires.authorgroup
            lst.append(1)
        except:
            lst.append(0)
    print(lst)
    t1 = time.time()
    print('expected single-thread time cost per record is: ' + str((t1-t0)/10.0))
    # we expect 121 seconds cost for 100 entries

print('done')

# it only took 20 seconds to do 1000 records
# that is 50 per second
# or a speed increase of factor 50x !
# but are they the same?
# it has no value if it is incorrect right
#
# + it will eat api-keys, so add a shuffler (tuto on pybliometrics page)
# + does it corrupt due api-rate-limits pushing out errors during overload?
# let's check this at a time we have a lot of hours
#
# first analysis shows almost 80% nulls

qq=1
qq=qq+1



# pickle.loads((res.scopus_abstract_obje[0])).authorgroup
# pickle.loads((x)).authorgroup
# res.scopus_abstract_obje.apply(lambda x: pickle.loads(x).authorgroup)
# fullres.scopus_abstract_obje.apply(lambda x: pickle.loads(x).authorgroup).to_csv(r'G:\UBVU\Data_RI\raw data algemeen\api_caches\test.csv')

fullres.scopus_abstract_obje.apply(lambda x: pickle.loads(x).authorgroup).to_csv(r'G:\UBVU\Data_RI\raw data algemeen\api_caches\test.csv')

# there are some sanitization issues, but overall it looks filled and good
#
# That means it works correctly right now! amazing!
# I assume it will remain to work well for records up to 6k, and if not we have retries to check if it failed due that
# and then just rerun!





