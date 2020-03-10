
import pandas as pd
import numpy as np
import time
from static import PATH_START_PERSONAL
from pybliometrics.scopus import ScopusSearch

df_orig = pd.read_excel(PATH_START_PERSONAL + '/arjan1.xlsx')
df_dois = pd.DataFrame(df_orig.DOI)
df_dois = df_dois.dropna()

bits = 10  # not dynamic yet
stepsize = int(np.ceil(len(df_dois) / bits)+1)

df_total = pd.DataFrame()
for cur_bit in np.arange(0, bits):
    print('-------')
    print(cur_bit)

    df_dois_CUR = df_dois.iloc[stepsize*cur_bit: stepsize*(cur_bit+1), :]

    doi_list_cur = df_dois_CUR['DOI'].to_list()
    cur_query = "DOI( " + " ) OR DOI( ".join(doi_list_cur) + " ) "

    if len(df_dois_CUR) > 0:
        t0 = time.time()
        fullres = pd.DataFrame(ScopusSearch(cur_query, download=True, refresh=True).results)
        t1 = time.time()
        print(t1-t0)

        df_total = df_total.append(fullres)

# backmerge it first
df_total = df_total.drop_duplicates(subset='doi')
df_export = df_orig.merge(df_total, left_on='DOI', right_on='doi', how='left')

df_export.to_csv(PATH_START_PERSONAL + 'arjan.csv')
df_export.to_excel(PATH_START_PERSONAL + 'arjan.xlsx')
