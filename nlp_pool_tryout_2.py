### code to run the multiprocessing code in pycharm
### remember: we need to be in the main to run the multiprocessing in python...





### DO NOT EDIT: MIGRATION UNDERWAY TOT NONMULTIPOOL> KEEP AS LEGACY





import cProfile  # remove later


from static import PATH_START, PATH_START_PERSONAL
from static import PATH_START_SERVER , PATH_START_PERSONAL_SERVER
import pandas as pd
import numpy as np
from multiprocessing import Pool
from nlp_functions import get_jaccard_sim
from nlp_functions import stack_titles
import time
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re
import html
sw = stopwords.words('english')

# settings (this you may edit, afterwards, just press run)
#
run_on_server = True
cap_pure_title_count = False  # this turns on test-mode with few samples
cap_scopus_title_count = False  # this turns on test-mode with few samples
lcap = 100  # 3000
rcap = lcap  # !
do_save = False
#
#
#
#
# outdated settings
run_entire_set = False  # leave False, not developed
run_subset_merged_only = False  # leave False, is done now







## helper functions
# helper functions
def remove_punctuation(text):
    '''a function for removing punctuation'''
    import string
    # replacing the punctuations with no space,
    # which in effect deletes the punctuation marks
    translator = str.maketrans('', '', string.punctuation)
    # return the text stripped of punctuation marks
    return text.translate(translator)

def remove_numbers(text):
    import string
    translator = str.maketrans('', '', '0123456789')
    return text.translate(translator)


def remove_stopwords_and_lower(text):
    '''a function for removing the stopword'''
    # removing the stop words and lowercasing the selected words
    text = [word.lower() for word in text.split() if word.lower() not in sw]
    # joining the list of words with space separator
    return " ".join(text)

def comma_space_fix(text):
    return (text
            .replace(", ", ",")
            .replace(",", ", ")
            .replace(". ", ".")
            .replace(".", ". ")
            .replace("; ", ";")
            .replace(";", "; "))  # this makes both ",x" and ", x" ", x"

def remove_html_unsafe(text):
    """
    This removes html stuff in order to avoid issues with subsequent NLP tools
    This is not a sanitation procedure and will not help with malicious html parts
    :param text: any text as string
    :return: processed text as string
    """
    tag_re = re.compile(r'(<!--.*?-->|<[^>]*>)')
    # Remove well-formed tags, fixing mistakes by legitimate users
    no_tags = tag_re.sub('', text)
    # Clean up anything else by escaping
    return html.escape(no_tags)

def pre_process(text, do_fill_empty=True, do_remove_html=False):
    """
    Just a method-chain turned into a function on vectorized level
    :param text: must be series of text
    :param do_fill_empty: whether you want to fill empty
    :param do_remove_html: whether you want to remove html (unsafe)
    :return:
    """
    if do_fill_empty:
        text = text.apply(do_fill_empty)
    if do_remove_html:
        text = text.apply(remove_html_unsafe)
    text = (text
            .apply(comma_space_fix)
            .apply(remove_punctuation)
            .apply(remove_numbers)
            .apply(remove_stopwords_and_lower)
            )
    return text




# resulting settings (do not edit from here unless you are developing)
#
#
use_split_types_routine = False  # turn off until team decides to add it
if run_on_server:
    print('server settings loaded')
    start_path = PATH_START_SERVER + r'\code speedup test data'
    out_path = PATH_START_SERVER + r'\code speedup test data\nlp2_result'
    num_workers = 4
else:
    print('local settings loaded')
    start_path = PATH_START +r'\raw data algemeen\code speedup test data'
    out_path = PATH_START +r'\raw data algemeen\code speedup test data\nlp2_result_fast'
    num_workers = 10  # change to your own specs/wishes as desired (100% CPU will claim the entire machine pretty much)

# get the data
#

if run_subset_merged_only:
    print('overriding run_entire_set and loading subset_merged')
    left_raw = pd.read_pickle(start_path + r'\df_combined_2018.pkl')
    right_raw = pd.read_pickle(start_path + r'\df_combined_2018.pkl')
    left_raw = left_raw[left_raw['merge_source'] == 'both']
    right_raw = right_raw[right_raw['merge_source'] == 'both']
else:
    if run_entire_set:
        print('running for entire set')
        #
        #
        # As much as I would love to run it for the entire set to evaluate the performance,
        # that performance metric is skewed because EID-matched stuff are usually English
        # and DOI-matching is also usually English and papers and also small share
        # so I will opt to leave this functionality as it is, and focus on the production-role:
        # that is, to merge the unmerged set
        # for further speed-up of development + in order to allow users to postpone the computation-heavy STM,
        # I will develop the STM as a post-processing step
        # getting this to production will be a lot of work, because it is not a simple .merge
        # I will need helper variables and splitters, but let's tackle it one step at a time
        #
        #
        print('invalid option: run_entire_set is not developed')
        #
        ###left_raw = pd.read_pickle(start_path + r'\ XXX .pkl')
        ###right_raw = pd.read_pickle(start_path + r'\ XXX .pkl')
        # leave the dutch and spanish and everything in it for now
        #
    else:
        print('running for unmerged set for all languages')
        ###left_raw = pd.read_pickle(start_path + r'\df_merge_candidate_C.pkl')  # equivalent to df_combined/ms=pure
        ###right_raw = pd.read_pickle(start_path + r'\df_Sunmerged2.pkl')  # equivalent to df_combined/ms=scopus
        left_raw = pd.read_csv(start_path + r'\df_unmerged_pure_pipeline_dec_2019.csv')  # equivalent to df_combined/ms=pure
        right_raw = pd.read_csv(start_path + r'\df_unmerged_scopus_pipeline_dec_2019.csv')  # equivalent to df_combined/ms=scopus
        # left_raw = left_raw[left_raw['Original language'] == 'English']


if cap_scopus_title_count:

    print('capping scopus records')
    right_raw = right_raw.iloc[0:rcap,:]
    print(rcap)


#
# either way, get ids
# we made a special id named post_merge_id for this purpose
# this will bug out old datasets but OK
# so you don't need to run code at all : )
#
###left_raw['left_id'] = left_raw.post_merge_id
###right_raw['right_id'] = right_raw.post_merge_id
#
#
# enrich with type-category:
# type-A: Journal // Contribution to Journal - Article or Contribution to Journal - Review article
# type-B: all the rest // al the rest
# first a check [care with pure or scopus changing stuff randomly]
if not(set(right_raw.aggregationType.unique()).issubset(set(['Journal', 'Book', 'Conference Proceeding', 'Book Series', np.nan]))):
    print('hey! scopus returned some aggregation types we do not recognize and will mess up our type categorization!')
if not(set(left_raw.Type.unique()).issubset(set(['Contribution to Journal - Article',
                                           'Contribution to Journal - Review article',
                                           'Chapter in Book / Report / Conference proceeding - Conference contribution',
                                           'Chapter in Book / Report / Conference proceeding - Chapter',
                                           'Book / Report - Book',
                                           'Book / Report - Report',
                                           'Contribution to Conference - Paper']))):
    print('hey! pure returned some aggregation types we do not recognize and will mess up our type categorization!')
#
#
right_raw['is_journal_type'] = right_raw.aggregationType.isin(['Journal'])
left_raw['is_journal_type'] = left_raw.Type.isin(['Contribution to Journal - Article',
                                                  'Contribution to Journal - Review article'])
#
# get left and right
left = left_raw[['Title of the contribution in original language',
                 'Subtitle of the contribution in original language',
                 'pub_uuid',
                 'post_merge_id',
                 'is_journal_type',
                 'pure_year']] \
    .dropna(subset=['Title of the contribution in original language'])
right = right_raw[['title', 'eid', 'post_merge_id', 'is_journal_type', 'scopus_year']].dropna(subset=['title'])
#

# process the text
do_fill_empty = False
do_remove_html = True
left['txt_raw'] = left.apply(stack_titles, axis=1)
left['txt'] = pre_process(text=left.txt_raw,
                          do_fill_empty=do_fill_empty,  # for nlp2 always
                          do_remove_html=do_remove_html)
right['txt'] = pre_process(right['title'],
                          do_fill_empty=do_fill_empty,  # for nlp2 always
                          do_remove_html=do_remove_html)


## the NLP functions everything is based on, I place them here because I cannot pull them due py-mp limits
#
def my_nlp2(x):
    """
    nlp function to test on (uses jaccard). DOES USE PREPROCESSING FROM NLP : )
    :param x: an index, data is pulled from elsewhere
    :return: a dictionary to append to a dataframe outside of the worker
    """
    df_js = pd.DataFrame()
    left_title = left.txt.iloc[x]  # one left title per worker # SUBOPTIMAL!
    left_title_raw = left.txt_raw.iloc[x]
    left_post_merge_id = left['post_merge_id'].iloc[x]
    left_pure_year = left['pure_year'].iloc[x]

    if use_split_types_routine:
        # then only use the subset of right where the is_journal_type matches with the current left[x]
        # this prevents matching different supertypes : )
        left_cur_is_journal_type = left.is_journal_type.iloc[x]
        right_subset = right[right.is_journal_type == left_cur_is_journal_type]
    else:
        right_subset = right
        left_cur_is_journal_type = None
        right_cur_is_journal_type = None


    for right_index in right_subset.index:
        right_title = right_subset.loc[right_index, 'txt']
        right_title_raw = right_subset.loc[right_index, 'title']
        right_post_merge_id = right_subset.loc[right_index, 'post_merge_id']
        right_cur_is_journal_type = right_subset.loc[right_index, 'is_journal_type']
        right_scopus_year = right_subset.loc[right_index, 'scopus_year']
        #
        score = get_jaccard_sim(left_title.split(' '), right_title.split(' '))
        df_js = df_js.append({'pure_title_raw': left_title_raw,
                              'pure_title': left_title,
                              'scopus_title': right_title,
                              'scopus_title_raw': right_title_raw,
                              'score': score,
                              'left_post_merge_id': left_post_merge_id,
                              'right_post_merge_id': right_post_merge_id,
                              'left_journal': left_cur_is_journal_type,
                              'right_journal': right_cur_is_journal_type,
                              'pure_year': left_pure_year,
                              'scopus_year': right_scopus_year
                              },
                             ignore_index=True)
    # I don't know how to return a full df via workers, maybe return a dict with top winner only and stop
    # later we can go to top 3 or w/e
    # goal is to test for speedup now anyway
    winner = df_js.sort_values('score', ascending=False).iloc[0].to_dict()  # cut communication to top-one+score only
    return winner


def my_nlp3(x):
    """
    nlp function to test on (uses jaccard). DOES USE PREPROCESSING FROM NLP : )
    :param x: an index, data is pulled from elsewhere
    :return: a dictionary to append to a dataframe outside of the worker
    """


    for cn in np.arange(0, rcap):
        left_title = "Going Concern Opinions and Management's Forward Looking Disclosures: Evidence from the MD&A" + x
        right_title = "Relation between duration of the prodromal phase and renal damage in ANCA-associated vasculitis" + x
        score = get_jaccard_sim(left_title.split(' '), right_title.split(' '))
        #score = get_jaccard_sim("Going Concern Opinions and Management's Forward Looking Disclosures: Evidence from the MD&A".split(' '),
        #                        "Relation between duration of the prodromal phase and renal damage in ANCA-associated vasculitis".split(' '))


    #winner = score
    return 0  # winner


def my_nlp4(x):
    """
    nlp function to test on (uses jaccard). DOES USE PREPROCESSING FROM NLP : )
    :param x: an index, data is pulled from elsewhere
    :return: a dictionary to append to a dataframe outside of the worker
    
    speedup-update: the x routine is nonsense, cut commcost with dict-pass
    """

    winner = {}

    ###df_js = pd.DataFrame()
    left_title = left.txt.iloc[x]  # one left title per worker # SUBOPTIMAL!
    left_title_raw = left.txt_raw.iloc[x]
    left_post_merge_id = left['post_merge_id'].iloc[x]
    left_pure_year = left['pure_year'].iloc[x]

    if use_split_types_routine:
        # then only use the subset of right where the is_journal_type matches with the current left[x]
        # this prevents matching different supertypes : )
        left_cur_is_journal_type = left.is_journal_type.iloc[x]
        right_subset = right[right.is_journal_type == left_cur_is_journal_type]
    else:
        right_subset = right
        left_cur_is_journal_type = None
        right_cur_is_journal_type = None

    best_score = -1  # will overwrite immediately as intended

    for right_index, right_title in zip(right_subset.index, right_subset.txt):


    ####for right_index in right_subset.index:
    ####    right_title = right_subset.loc[right_index, 'txt']
        ###right_title_raw =
        ###right_post_merge_id =
        ###right_cur_is_journal_type =
        ###right_scopus_year =
        #
        score = get_jaccard_sim(left_title.split(' '), right_title.split(' '))

        if score > best_score:
            # new best score
            best_score = score
            # overwrite results
            best_left_title_raw = left_title_raw
            best_left_title = left_title
            best_right_title = right_title
            best_right_title_raw = right_subset.loc[right_index, 'title']
            best_left_post_merge_id = left_post_merge_id
            best_right_post_merge_id = right_subset.loc[right_index, 'post_merge_id']
            best_left_cur_is_journal_type = left_cur_is_journal_type
            best_right_cur_is_journal_type = right_subset.loc[right_index, 'is_journal_type']
            best_left_pure_year = left_pure_year
            best_right_scopus_year = right_subset.loc[right_index, 'scopus_year']

    winner = {'pure_title_raw': best_left_title_raw,
                              'pure_title': best_left_title,
                              'scopus_title': best_right_title,
                              'scopus_title_raw': best_right_title_raw,
                              'score': best_score,
                              'left_post_merge_id': best_left_post_merge_id,
                              'right_post_merge_id': best_right_post_merge_id,
                              'left_journal': best_left_cur_is_journal_type,
                              'right_journal': best_right_cur_is_journal_type,
                              'pure_year': best_left_pure_year,
                              'scopus_year': best_right_scopus_year
                              }
    # I don't know how to return a full df via workers, maybe return a dict with top winner only and stop
    # later we can go to top 3 or w/e
    # goal is to test for speedup now anyway
    #winner = df_js.sort_values('score', ascending=False).iloc[0].to_dict()  # cut communication to top-one+score only
    #print('nlp4')
    return winner



if cap_pure_title_count:
    cap = lcap
    #print('capping titles to compare count at ' + str(cap))
    print(lcap)
else:
    cap = len(left)
list_x = list(np.arange(0, cap))  # do everything # len(left)
#
#

#print(list_x)
#print(len(list_x))
# actually do the my_nlp2 task
#
start = time.time()
if False:  # tester
    if True:

        prof = False
        if prof:
            pr = cProfile.Profile()
            pr.enable()

        for cur_x in list_x:
            #my_nlp3('a')   # 100x100: 0.027
            my_nlp4(1)      # 0.4 or 20x slower, but still much better
            # str(np.round(np.random.rand(1),4))

        if prof:
            pr.disable()
            pr.print_stats(sort="tottime")  # calls
            input('waiting')



    else:
        if do_save:
            if __name__ == '__main__':
                with Pool(num_workers) as p:
                    df_winners = pd.DataFrame(p.map(my_nlp3, list_x,))
                    #print(df_winners.head())
                    df_winners.to_excel(out_path + '.xlsx')
                    df_winners.to_csv(out_path + '.csv')
        else:
            if __name__ == '__main__':
                with Pool(num_workers) as p:
                    df_winners = pd.DataFrame(p.map(my_nlp3, list_x, ))
                    #print(df_winners.head())
                    # df_winners.to_excel(out_path + '.xlsx')
                    # df_winners.to_csv(out_path + '.csv')
else:
    # PRODUCTION-FORM
    use_new_nlp4 = True
    if use_new_nlp4:
        use_multi = (len(list_x)*len(list_x) > 400000)  # uses single if dataset is small to avoid startup time costs 4s
        if use_multi:
            # it is
            if do_save:
                if __name__ == '__main__':
                    with Pool(num_workers) as p:
                        df_winners = pd.DataFrame(p.map(my_nlp4, list_x,))
                        #print(df_winners.head())
                        df_winners.to_excel(out_path + '.xlsx')
                        df_winners.to_csv(out_path + '.csv')
            else:
                if __name__ == '__main__':
                    with Pool(num_workers) as p:
                        df_winners = pd.DataFrame(p.map(my_nlp4, list_x, ))
                        #print(df_winners.head())
                        # df_winners.to_excel(out_path + '.xlsx')
                        # df_winners.to_csv(out_path + '.csv')
        else:
            df_winners = pd.DataFrame()
            for cur_x in list_x:
                df_winners = df_winners.append(my_nlp4(cur_x), ignore_index=True)
    else:
        if do_save:
            if __name__ == '__main__':
                with Pool(num_workers) as p:
                    df_winners = pd.DataFrame(p.map(my_nlp2, list_x,))
                    #print(df_winners.head())
                    df_winners.to_excel(out_path + '.xlsx')
                    df_winners.to_csv(out_path + '.csv')
        else:
            if __name__ == '__main__':
                with Pool(num_workers) as p:
                    df_winners = pd.DataFrame(p.map(my_nlp2, list_x, ))
                    #print(df_winners.head())
                    # df_winners.to_excel(out_path + '.xlsx')
                    # df_winners.to_csv(out_path + '.csv')

#time.sleep(0.001)
end = time.time()
print('Time taken in seconds -', np.round(end - start, 8))
#print('num_workers is ' + str(num_workers))


# final updates:
# my_nlp4 is much faster and verified now, so please use it !
# only difference is nlp4 does not overwrite for equal scores to save comp time (but is equally correct)
# should work one-shot on server if paths are correct






