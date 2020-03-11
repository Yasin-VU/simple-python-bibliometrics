# aggregates pure+scopus combined table columns into combined columns with dynamic rules
# PS: do not hierarchy-aggregate verboses as the non-nan strings will not allow overwriting

# I copied all the code over, but it needs a lot of refactoring before it can work in a reliable manner (!)
# for example, the column names changed with amsco...
# and all the post-fixes: they need to be split up and simplified (even if it means longer computation times)
#
# good news is that this py file is the final py file before the oadash2019 data is ready for validation&use


import pandas as pd
import numpy as np
import sys
from nlp_functions import stack_titles
from open_deploy_draft import predict_faculty
from core_functions import fn_cats
from core_functions import add_year_and_month
from core_functions import get_scopus_abstract_info
from core_functions import get_first_chosen_affiliation_author
from core_functions import add_unpaywall_columns
from core_functions import my_timestamp
from core_functions import add_deal_info
from nlp_functions import corresponding_author_functions


# settings
#
do_save = True
#
start_path = r'G:\UBVU\Data_RI\raw data algemeen\oa2019map'
df = pd.read_csv(start_path + r'\merged_data\refactor_test.csv')
#
# post-fix settings
chosen_affid = ["60008734","60029124","60012443","60109852","60026698","60013779","60032886","60000614",
                "60030550","60013243","60026220","60001997"]  # I added 60001997 and thus I added VUMC
vu_afids = chosen_affid
path_vsnu_afids = r'G:\UBVU\Data_RI\raw data algemeen\afids_vsnu_nonfin.csv'
all_vsnu_sdg_afids = pd.read_csv(path_vsnu_afids).iloc[:,1].astype('str').to_list()
#
path_deals = r'G:\UBVU\Data_RI\raw data algemeen\apcdeals.csv'
path_isn = r'G:\UBVU\Data_RI\raw data algemeen\ISN_ISSN.csv'
#


# example aggregation-function
def agg_cols_pass_single(cols, new_name):
    # cols is a pandas dataframe with multiple columns, ordered in hierarchy
    # this agg_cols directly passes first column, nothing else
    if cols is None:
        # minor robustness
        return None
    if len(cols.shape) == 1:  # series
        return cols.rename(new_name)  # as is, do rename
    else:
        res = cols.iloc[:, 0]  # index 0
        # for ii in np.arange(cols.shape[1]-1)+1:  # jump index 0, do all others
        #    res = res.fillna(cols.iloc[:,ii])
        # res = res.rename(new_name)
        return res  # extremely fast vectorized code


# another example aggregation-function
def agg_cols_hierarchical(cols, new_name):
    # cols is a pandas dataframe with multiple columns, ordered in hierarchy
    if cols is None:
        # minor robustness
        return None
    if len(cols.shape) == 1:  # series
        return cols.rename(new_name)  # as is, do rename
    else:
        res = cols.iloc[:, 0]  # index 0
        for ii in np.arange(cols.shape[1] - 1) + 1:  # jump index 0, do all others
            res = res.fillna(cols.iloc[:, ii])
        res = res.rename(new_name)
        return res  # extremely fast vectorized code


def multi_aggregator(agg_reqs):
    # input must be a list containing lists of an aggregation-function, a string for the new column name, and stacked columns
    # the stacked columns must be a pandas dataframe
    res = pd.DataFrame()
    for cur_agg in agg_reqs:
        # unpack
        agg_fn = cur_agg[0]
        new_name = cur_agg[1]
        cols = cur_agg[2]  # stacked as a dataframe
        subres = agg_fn(cols, new_name)
        res = pd.concat([res, subres], axis=1)
    return res

# make missing columns
df['pure_title_plus_subtitle'] = df.apply(stack_titles, axis=1)
# also make the partly empty oa cols so distilling will work as intended, they are cheap operations
df['oa_color_category'] = df.open_access_color.apply(fn_cats)
df['upw_oa_color_verbose'] = df['upw_oa_color'].apply(lambda x: 'unknown' if x is np.nan else x)  # double-code !

# I checked all these columns manually one by one, they seem to work for amsco too which is nice
#
# default rule_set
# v2: always scopus first except when: ff_match: done
# this list tells the computer which columns to aggregate with which rule and which name
my_agg_reqs = [
    [agg_cols_hierarchical, 'doi', df[['doi','DOI','upw_doi']]],  # scopus first
    [agg_cols_hierarchical, 'issn', df[['issn', 'Journal > Journal[1]:ISSNs']]],  # scopus first
    [agg_cols_hierarchical, 'journal_name', df[['journal_name', 'Journal > Journal[1]:Titles']]],  # scopus first
    [agg_cols_hierarchical, 'original_language', df[['Original language',]]],
    [agg_cols_hierarchical, 'title', df[['title','pure_title_plus_subtitle']]],  # ! scopus first here
    [agg_cols_hierarchical, 'type', df[['aggregationType', 'Type']]], # ! scopus first here, keep in mind in power bi filters
    [agg_cols_hierarchical, 'uuid', df[['UUID','pub_uuid']]],
    [agg_cols_hierarchical, 'workflow_step', df[['Workflow > Step',]]],
    [agg_cols_hierarchical, 'abstract_text_clean', df[['abstract_text_clean',]]],
    [agg_cols_hierarchical, 'deal_discount_verbose', df[['deal_discount_verbose',]] ],  # verbose will never reach column2 !
    [agg_cols_hierarchical, 'deal_name', df[['deal_name',]]],
    [agg_cols_hierarchical, 'deal_owner_raw', df[['deal_owner_raw',]]],
    [agg_cols_hierarchical, 'deal_scope', df[['deal_scope',]]],
    [agg_cols_hierarchical, 'eIssn', df[['eIssn',]]],
    [agg_cols_hierarchical, 'eid', df[['eid',]]],
    [agg_cols_hierarchical, 'faculty_(matched)', df[['ff_match','faculty_(matched)',]]],  # some from pure, some from scopus !
    [agg_cols_hierarchical, 'first_VU_author', df[['first_VU_author',]]],
    [agg_cols_hierarchical, 'fulltext_free_url', df[['fulltext_free_url','upw_free_fulltext_url']]],
    [agg_cols_hierarchical, 'fund_sponsor', df[['fund_sponsor',]]],
    [agg_cols_hierarchical, 'is_corresponding_author_a_ukb_author', df[['is_corresponding_author_a_ukb_author',]]],
    [agg_cols_hierarchical, 'is_corresponding_author_a_vu_author', df[['is_corresponding_author_a_vu_author',]]],
    [agg_cols_hierarchical, 'is_free_to_read', df[['is_free_to_read','upw_is_free_to_read']]],
    [agg_cols_hierarchical, 'license', df[['license','upw_license']]],
    [agg_cols_hierarchical, 'merge_source', df[['merge_source',]]],
    [agg_cols_hierarchical, 'open_access_color', df[['open_access_color','upw_oa_color']]],
    [agg_cols_hierarchical, 'year', df[['pure_year','scopus_year']]],
    [agg_cols_hierarchical, 'is_a_subscription_journal', df[['is_a_subscription_journal','upw_is_subscription_journal']]],
    [agg_cols_hierarchical, 'upw_oa_color_verbose', df[['upw_oa_color_verbose']]],  # verbose will never reach column2 !
    [agg_cols_hierarchical, 'oa_color_category', df[['oa_color_category','upw_oa_color_category']]],
    [agg_cols_hierarchical, 'vu_contact_person', df[['vu_contact_person']]],
    [agg_cols_hierarchical, 'type_contains_book', df[['type_contains_book']]],
    ]

# run the multi-aggregator to obtain the distilled dataframe!
df_distil = multi_aggregator(agg_reqs=my_agg_reqs)  # reduces column count from 102 to 30 : )
# now with 1 extra index_doi column only, verified

# fill the book info on scopus
df_distil.loc[df_distil.merge_source=='scopus', 'type_contains_book'] = df_distil[df_distil.merge_source=='scopus'].type.isin(['Book', 'Book Series'])



### simplify the code below: just redo everything instead of post-fixing 10 times

add_cols2 = [#'deal_discount_verbose',
             #'deal_name',
             #'deal_owner_raw',
             #'deal_scope',
             #'first_VU_author',
             #'is_corresponding_author_a_ukb_author',
             #'is_corresponding_author_a_vu_author',
             #'vu_contact_person'
             ]
upw_cols = [  #### 'doi',  # not this one
    'doi_lowercase',
    'orig_doi',
    'upw_doi',
    'upw_doi_lowercase',
    'upw_error',
    'upw_error_message',
    'upw_free_fulltext_url',
    'upw_is_boai_license',
    'upw_is_free_to_read',
    'upw_is_subscription_journal',
    'upw_license',
    'upw_oa_color']

# ISSN cleaning: no '-' (add checks here btw)
if not (df_distil.loc[df_distil.merge_source=='pure', 'issn'].str.len().median() == 8+1):
    print('the issn cleaning may be going wrong, please check')
df_distil.loc[df_distil.merge_source=='pure', 'issn'] = (df_distil
                                                         .loc[df_distil.merge_source=='pure', 'issn']
                                                         .apply(lambda x: np.nan if x is np.nan else x[0:4] + x[5:8]))
# df_distil.loc[df_distil.merge_source=='pure', 'issn'].str.len().median()
# df_distil.loc[df_distil.merge_source=='pure', 'issn'][df_distil.loc[df_distil.merge_source=='pure', 'issn'].str.len() > 8]
#
# hey: there are comma-delimited multiple issns in 343 records in pure... some with 3 issns
# that will definitely mess up the deals
# future action: !
# for now, let's just cut everything after the first and make the ASSUMPTION that a first ISSN is sufficient for deals







# adds nan columns
df_distil.loc[df_distil.merge_source=='pure', 'first_VU_author'] = np.nan
df_distil.loc[df_distil.merge_source=='pure', 'is_corresponding_author_a_ukb_author'] = np.nan
df_distil.loc[df_distil.merge_source=='pure', 'is_corresponding_author_a_vu_author'] = np.nan
df_distil.loc[df_distil.merge_source=='pure', 'vu_contact_person'] = np.nan


# add deal info
#
# pure part, no deal coluns
# (df_distil.loc[df_distil.merge_source=='pure', :].drop(columns=[w for w in list(df_distil) if w[0:4] == 'deal']))
#
# the basic add_deal_info is not sufficient: because we post-processed it, and need to replicate it here
# PLEASE: refactor this ! make sure pure is pre-processed with this and then pushed through the distiller !
cur_res = add_deal_info(path_deals=path_deals, path_isn=path_isn, df_b=(df_distil.loc[df_distil.merge_source=='pure', :].drop(columns=[w for w in list(df_distil) if w[0:4] == 'deal']))).rename(columns={
    'deal_owner': 'deal_owner_raw',
    'deal_owner_verbose': 'deal_scope'
}).drop(columns=['deal_ISN',
  'deal_journal_title',
  'deal_discount',
  'deal_modified',
  'deal_ISSN',
  'deal_ISSN_short',
])
df_distil_p = df_distil[df_distil.merge_source == 'pure']
df_distil_np = df_distil[~(df_distil.merge_source == 'pure')]
df_distil_p = cur_res.copy()
df_distil = df_distil_p.append(df_distil_np)
#
# verified now I guess, though pure-only gave zero deals but ok

# remains unpaywall
distil_upw_cols = ['fulltext_free_url',
                     'is_free_to_read',
                     'license',
                     'open_access_color',
                     'is_a_subscription_journal',
                     'upw_oa_color_verbose',
                     'oa_color_category']
base_upw_cols = [ 'id_lowercase',
                     'orig_doi',
                     'own_doi_lowercase',
                     'upw_doi',
                     'upw_doi_lowercase',
                     'upw_error',
                     'upw_error_message',
                     'upw_free_fulltext_url',
                     'upw_is_boai_license',
                     'upw_is_free_to_read',
                     'upw_is_subscription_journal',
                     'upw_license',
                     'upw_oa_color']
#
#
# I will keep it simple and just redo all rows, not just the pure-only
# we have multi-thread so it should be fast enough
#
print(df_distil.merge_source.unique())
import time
t0 = time.time()
df_distil = add_unpaywall_columns(df_distil.drop(columns=distil_upw_cols)).rename(columns={'upw_free_fulltext_url': 'fulltext_free_url',
                         'upw_is_free_to_read': 'is_free_to_read',
                         'upw_license': 'license',
                         'upw_oa_color': 'open_access_color',
                         'upw_is_subscription_journal': 'is_a_subscription_journal'
                         }).drop(columns=[   'id_lowercase',
                                             'orig_doi',
                                             'own_doi_lowercase',
                                             'upw_doi',
                                             'upw_doi_lowercase',
                                             'upw_error',
                                             'upw_error_message',
                                             'upw_is_boai_license',
                                             ])
df_distil['upw_oa_color_category'] = df_distil.open_access_color.apply(fn_cats)
df_distil['upw_oa_color_verbose'] = df_distil.open_access_color.apply(lambda x: 'unknown' if x is np.nan else x)
print('upw done')
print(df_distil.merge_source.unique())
t1 = time.time()
print(t1-t0)
#
# this should fix unpaywall


# check. chk upw results: seems OK: continue (remember that PURE has very little DOI so unknowns are very common
# 2. fac-unknown fix



qq=1
qq+=1



# ASSUMPTION: this model is good enough : )

### fac-unknown fixes ###
# get the model
import pickle
export_name = 'production_model_1'
filename = 'C:/Users/yasin/Desktop/ML X Y 2/' + export_name + '.pkl'
# load the model from disk
model = pickle.load(open(filename, 'rb'))
print(model)
filename_tfidf = 'C:/Users/yasin/Desktop/ML X Y 2/tfidf_' + export_name + '.pkl'
# load the model from disk
tfidf = pickle.load(open(filename_tfidf, 'rb'))
print(tfidf)
#
df_distil_rich_no_fac_unknown = df_distil.copy()  # removed rich in refactor !
df_distil_rich_no_fac_unknown = df_distil_rich_no_fac_unknown.rename(columns={'doi': 'DOI'})
df_distil_rich_no_fac_unknown['first_VU_author_raw_organization_info'] = df_distil_rich_no_fac_unknown[
    'faculty_(matched)']  # what? why?!
#
res = predict_faculty(
    df_distil_rich_no_fac_unknown.loc[df_distil_rich_no_fac_unknown['faculty_(matched)'].isnull(), :],
    model,
    tfidf).copy()

# df_distil_rich_no_fac_unknown[df_distil_rich_no_fac_unknown['faculty_(matched)'].isnull()]

df_distil_rich_no_fac_unknown['fac_unknown'] = df_distil_rich_no_fac_unknown['faculty_(matched)'].isnull()

# df_distil_rich_no_fac_unknown[df_distil_rich_no_fac_unknown['fac_unknown']].loc[:, 'faculty_(matched)'] = res
pa = df_distil_rich_no_fac_unknown[df_distil_rich_no_fac_unknown['fac_unknown']]
pb = df_distil_rich_no_fac_unknown[~(df_distil_rich_no_fac_unknown['fac_unknown'])]
pa.loc[:, ['faculty_(matched)']] = res
df_distil_rich_no_fac_unknown = pa.append(pb)
#
# fixing pandas is tiresome but now it works

df_distil_rich_no_fac_unknown = df_distil_rich_no_fac_unknown.rename(columns={'DOI': 'doi'})
df_distil_rich_no_fac_unknown = df_distil_rich_no_fac_unknown.drop(columns=['first_VU_author_raw_organization_info'])
#
# post-fix 4
df_distil_rich_no_fac_unknown['faculty_(matched)'] = df_distil_rich_no_fac_unknown['faculty_(matched)'].apply(lambda x:
                                                                                                 'VUmc' if x == 'medical center' else x)




qq=1
qq+=1

#
if do_save:
    df_distil_rich_no_fac_unknown.to_csv(start_path + r'\RETRY_NEW_df_total_with_STM_2019_with_keuzemodel_faculty_filled.csv')
#
#

print('done with py file')







# below we do not intend to run, it just for reference, delete it after refactoring
if False:
    # don't worry about df2 renaming, your df does not use those

    add_cols2 = ['deal_discount_verbose',
                 'deal_name',
                 'deal_owner_raw',
                 'deal_scope',
                 'first_VU_author',
                 'is_corresponding_author_a_ukb_author',
                 'is_corresponding_author_a_vu_author',
                 'vu_contact_person']
    upw_cols = [#'doi',  # not this one
     'doi_lowercase',
     'orig_doi',
     'upw_doi',
     'upw_doi_lowercase',
     'upw_error',
     'upw_error_message',
     'upw_free_fulltext_url',
     'upw_is_boai_license',
     'upw_is_free_to_read',
     'upw_is_subscription_journal',
     'upw_license',
     'upw_oa_color']
    df_distil_pure = df_distil_pure.drop(columns=list(set(upw_cols).intersection(set(list(df_distil_pure)))))


    # split for pure y/n, grab pure part, drop these cols, make these cols again, and rebuild df_distil
    # some can't be made yet due to missing functionality like corr_author:
    df_distil_pure['first_VU_author'] = np.nan
    df_distil_pure['is_corresponding_author_a_ukb_author'] = np.nan
    df_distil_pure['is_corresponding_author_a_vu_author'] = np.nan
    df_distil_pure['vu_contact_person'] = np.nan

    ############################################################


    ### post-fixes
    # next: tackle the issues with enrichment missing
    # we want these extra:
    add_cols = ['deal_discount_verbose',
                 'deal_name',
                 'deal_owner_raw',
                 'deal_scope',
                 'first_VU_author',
                 'is_corresponding_author_a_ukb_author',
                 'is_corresponding_author_a_vu_author',
                 'vu_contact_person']

    # ! we need to split, remove columns, apply, and re-append here
    # split
    df_distil_pure = df_distil[df_distil.merge_source=='pure']
    df_distil_not_pure = df_distil[~(df_distil.merge_source=='pure')]
    # remove columns
    df_distil_pure = df_distil_pure.drop(columns=add_cols)  # valid thanks to all columns starting with deal_ : )

    # add deal info
    #
    #
    df_distil_pure = add_deal_info(path_deals=path_deals, path_isn=path_isn, df_b=df_distil_pure)
    df_distil_pure = df_distil_pure.rename(columns={
     'deal_owner' : 'deal_owner_raw',
     'deal_owner_verbose' : 'deal_scope'
    })  # rename required for consistency  # not ideal form but OK
    #
    # some cleaning of unwanted columns: we need to match the original columns of df_distil
    valid_cols = set(list(df_distil_pure)).union(set(list(df_distil)))
    df_distil_pure = df_distil_pure.loc[:,valid_cols]
    df_distil_pure[['deal_discount_verbose',
                 'deal_name',
                 'deal_owner_raw',
                 'deal_scope']].head()

    # add corresponding author info
    # scopus routine will not work here because we have no abstract_object
    # is there any other way to find out? there should be
    # status right now is: it is inside pure, but we cannot get it out easily yet
    #                      we can access it through the pure API for recent years, but development is required then
    #                      hence I will leave this as a point of discussion for the next sprint session (has been e-mailed)
    #
    # (!) add empty cols to avoid columns mismatch during appending
    df_distil_pure['first_VU_author'] = np.nan
    df_distil_pure['is_corresponding_author_a_ukb_author'] = np.nan
    df_distil_pure['is_corresponding_author_a_vu_author'] = np.nan



    # add vu_contact_person: we don't have corresponding author, so let's grab second best
    # I found out our pure pipeline does not store any author info at all ever
    # I can postprocess it back in by matching on uuid
    # but this must be discussed at the sprintsession first because it would take time and I am not sure if we want it (prolly?)
    # also, it depends a lot on if you want corresponding author in it or not
    df_distil_pure['vu_contact_person'] = np.nan

    # checks
    df_distil_pure[['deal_name',
                 'deal_owner_raw']].isnull().mean(0)


    # extra unpaywall for pure-only
    # run unpaywall for the pure-only part of df_distil because we previously skipped it and we want unpaywall info !
    #
    # first remove the columns for the function
    upw_cols = [#'doi',  # not this one
     'doi_lowercase',
     'orig_doi',
     'upw_doi',
     'upw_doi_lowercase',
     'upw_error',
     'upw_error_message',
     'upw_free_fulltext_url',
     'upw_is_boai_license',
     'upw_is_free_to_read',
     'upw_is_subscription_journal',
     'upw_license',
     'upw_oa_color']
    df_distil_pure = df_distil_pure.drop(columns=list(set(upw_cols).intersection(set(list(df_distil_pure)))))
    df_distil_pure

    #len(df_distil_pure.doi.unique())  # this is not OK: we need a dupe-call preventtion  # update: code updated in core_fn
    df_distil_pure = add_unpaywall_columns(df_distil_pure, silent=False, cut_dupes=True)
    # the column naming is off and causing issues !!!



    drop_list_final_naming = ['license',
    'open_access_color',
    'is_a_subscription_journal',
    'is_free_to_read',
    'oa_color_category']


    upw_rename_dict = {
     'upw_is_free_to_read': 'is_free_to_read',
     'upw_is_subscription_journal': 'is_a_subscription_journal',
     'upw_license':'license',
     'upw_oa_color':'open_access_color',
    'upw_oa_color_category':'oa_color_category'  # issue fix
     ###'fulltext_free_url':'upw_free_fulltext_url' # no
    }
    upw_delete_cols = [ 'upw_doi',
                         'upw_doi_lowercase',
                         'upw_error',
                         'upw_error_message',
                         'upw_is_boai_license',
                             'upw_doi_lowercase',
                             'orig_doi',
                 'own_doi_lowercase','id_lowercase']

    # post-process (double-code, fix later)
    df_distil_pure['upw_oa_color_category'] = df_distil_pure.upw_oa_color.apply(fn_cats)
    df_distil_pure['upw_oa_color_verbose'] = df_distil_pure['upw_oa_color'].apply(lambda x: 'unknown' if x is np.nan else x)
    #
    # make matching
    df_distil_pure = df_distil_pure.drop(columns=drop_list_final_naming)
    df_distil_pure = df_distil_pure.rename(columns=upw_rename_dict)
    df_distil_pure = df_distil_pure.drop(columns=upw_delete_cols)

    # extra cleaning
    delete_cols = ['deal_modified',
     'deal_discount',
     'deal_journal_title',
     'deal_ISSN_short',
     'deal_ISN',
     'deal_ISSN',
     'upw_free_fulltext_url'
                  ]

    df_distil_pure = df_distil_pure.drop(columns=delete_cols)
    ##

    # now distil df_distil_pure back
    df_distil_rich = df_distil_not_pure.append(df_distil_pure)

    # FIX ISSUE UPW-1: just redo it, also makes it recent too...

    list(df_distil_rich)
    #
    df_distil_rich = df_distil_rich.drop(columns=list(set(upw_cols).intersection(set(list(df_distil_rich)))))
    df_distil_rich = add_unpaywall_columns(df_distil_rich, silent=False, cut_dupes=True)
    #
    #
    df_distil_rich['upw_oa_color_category'] = df_distil_rich.upw_oa_color.apply(fn_cats)
    df_distil_rich['upw_oa_color_verbose'] = df_distil_rich['upw_oa_color'].apply(lambda x: 'unknown' if x is np.nan else x)
    # make matching
    df_distil_rich = df_distil_rich.drop(columns=drop_list_final_naming)
    df_distil_rich = df_distil_rich.rename(columns=upw_rename_dict)
    df_distil_rich = df_distil_rich.drop(columns=upw_delete_cols)
    df_distil_rich = df_distil_rich.drop(columns=['upw_free_fulltext_url'])
    # this is why refactoring is important

    # now check it

    df_distil_rich['tmp_has_doi'] = ~df_distil_rich.doi.isnull()

    # soms krijg je None en dat hoort niet, post-fix dat
    df_distil_rich['upw_oa_color_verbose'] = df_distil_rich.upw_oa_color_verbose.apply(lambda x: x if pd.notnull(x) else 'unknown')

    # postfix 2
    df_distil_rich['open_access_color'].unique()
    df_distil_rich = df_distil_rich.drop(columns=['open_access_color'])  # avoid confusion

    # postfix 3
    df_distil_rich['upw_oa_color_category'] = df_distil_rich.upw_oa_color_verbose.apply(fn_cats)

    ### end of post-fixes

    if False:
        df_distil_rich.to_csv(start_path + r'\df_total_with_STM_2018_with_keuzemodel.csv')
        df_distil_rich[['title', 'merge_source']].sort_values(by='title').to_csv(start_path + r'\df_jordy.csv')



    ### fac-unknown fixes
    # get the model

    import pickle
    export_name = 'production_model_1'
    filename = 'C:/Users/yasin/Desktop/ML X Y 2/' + export_name + '.pkl'
    # load the model from disk
    model = pickle.load(open(filename, 'rb'))
    print(model)

    filename_tfidf = 'C:/Users/yasin/Desktop/ML X Y 2/tfidf_' + export_name + '.pkl'
    # load the model from disk
    tfidf = pickle.load(open(filename_tfidf, 'rb'))

    print(tfidf)

    df_distil_rich_no_fac_unknown = df_distil_rich.copy()
    df_distil_rich_no_fac_unknown = df_distil_rich_no_fac_unknown.rename(columns={'doi':'DOI'})
    df_distil_rich_no_fac_unknown['first_VU_author_raw_organization_info'] = df_distil_rich_no_fac_unknown['faculty_(matched)']

    res = predict_faculty(
        df_distil_rich_no_fac_unknown.loc[df_distil_rich_no_fac_unknown['faculty_(matched)'].isnull(),:],
        model,
        tfidf).copy()
    df_distil_rich_no_fac_unknown.loc[df_distil_rich_no_fac_unknown['faculty_(matched)'].isnull(),'faculty_(matched)'] = res

    df_distil_rich_no_fac_unknown = df_distil_rich_no_fac_unknown.rename(columns={'DOI':'doi'})
    df_distil_rich_no_fac_unknown = df_distil_rich_no_fac_unknown.drop(columns=['first_VU_author_raw_organization_info'])

    df_distil_rich_no_fac_unknown['faculty_(matched)']

    # post-fix 4
    #?????????


    df_distil_rich_no_fac_unknown['faculty_(matched)'] = df_distil_rich_no_fac_unknown['faculty_(matched)'].apply(lambda x:
             'VUmc' if x == 'medical center' else x)

    if False:
        df_distil_rich_no_fac_unknown.to_csv(start_path + r'\df_total_with_STM_2018_with_keuzemodel_faculty_filled.csv')

    # all affils
    dft = df[df.merge_source=='both']['Organisations > Organisational unit[1]'].head()
    from nlp_functions import faculty_finder
    org_info = pd.read_excel(r'G:\UBVU\Data_RI\raw data algemeen\vu_organogram_2.xlsx', skiprows=0)
    ff = faculty_finder(organizational_chart=org_info)
    dft.apply(lambda x: ff.match(x)['ff_match'])
    #
    def get_all_affils(row):
        lst = []
        for ii in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            colname = 'Organisations > Organisational unit[' + str(ii) + ']'
            res = ff.match(row[colname])['ff_match']
            if pd.notnull(res):
                lst.append(res)

        return lst  # set has less info, so keep it like this for now, we can always set later
    df['all_vu_affils'] = df.apply(get_all_affils, axis=1)
    # 7 s for 100 means 81*7 = 10 minutes for everything
    # just run it
    #
    if False:
        df.to_csv(start_path + r'\df_total_with_STM_2018_all_vu_affils_2.csv')
    dict_encode_fac = {0:'Faculty of Science',
                       1:'Faculty of Behavioural and Movement Sciences',
                       2:'medical center',
                       3:'Faculty of Social Sciences',
                       4:'School of Business and Economics',
                       5:'Faculty of Law',
                       6:'Faculty of Humanities',
                       7:'Faculty of Religion and Theology',
                       8:'ACTA',
                       #9:,
                      }  # matches encode_fac()
    # each will be yes/no

    for cur_fac, cur_fac_enc in zip(dict_encode_fac.values(), dict_encode_fac.keys()):
        df.loc[:,'has_fac_'+str(cur_fac_enc)] = df.all_vu_affils.apply(lambda x: np.nan if len(x)==0 else cur_fac in x )
    if False:
        df.to_csv(start_path + r'\df_total_with_STM_2018_all_vu_affils_2_X.csv')

    df_old = pd.read_csv(r'G:\UBVU\Data_RI\raw data algemeen\machine learning' + r'\merged_data\df_total.csv')
    df_old['all_vu_affils'] = df_old.apply(get_all_affils,axis=1)
    for cur_fac, cur_fac_enc in zip(dict_encode_fac.values(), dict_encode_fac.keys()):
        df_old.loc[:, 'has_fac_' + str(cur_fac_enc)] = df_old.all_vu_affils.apply(
            lambda x: np.nan if len(x) == 0 else cur_fac in x)
    # this may take half an hour
    if False:
        df_old.to_csv(r'G:\UBVU\Data_RI\raw data algemeen\machine learning' + r'\merged_data\df_total_all_affils_X.csv')
