# NLP functions
#
# this file contains reusable Natural Language Processing functions
# the file is separate to prevent unnecessary loading of NLP packages and isolate issues when they occur


#from static import PATH_START, PATH_START_PERSONAL
#from static import PATH_START_SERVER , PATH_START_PERSONAL_SERVER
import pandas as pd
import numpy as np
import editdistance  # third-party, I need a checker for this to default to regular or local levenschtein (care 0=best)
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re
import html
nltk.download('punkt', quiet=True)  # check for punkt silently

class SoftTitleMatcher:
    """
    I want all processing here and a non-multipool nonfork variant in here
    
    example code:
        stm = soft_title_matcher(silent=False)
    
    
    many conditions on columnnames...
    careful: cross-language title matching is not present
    looks fantastic, only some issues with 'response/correction to', but we can pick that up from the pure fields later
    """
    
    def __init__(self, silent=True):
        self.silent = silent
        
        # remember we have multiple routines here, including nlp4


    def preprocess_and_find_best_matches(self,
                                         left_raw,
                                         right_raw,
                                         out_path=None,
                                         do_save=False):
        """
        Here I want a nonfork version of nlp4 so I can integrate easily
        with the rest of the pipeline
        """

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

        # preprocess
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
        


        
        
        # wrapper
        list_x = list(np.arange(0, len(left)))
        df_winners = pd.DataFrame()
        for cur_x in list_x:


            left_title = left.txt.iloc[cur_x]  # one left title per worker # SUBOPTIMAL!
            left_title_raw = left.txt_raw.iloc[cur_x]
            left_post_merge_id = left['post_merge_id'].iloc[cur_x]
            left_pure_year = left['pure_year'].iloc[cur_x]


            df_winners = df_winners.append(self.find_best_match(left_title,
                                                                left_title_raw,
                                                                left_post_merge_id,
                                                                left_pure_year,
                                                                right
                                                                ), ignore_index=True)

        # save to csv
        # df_winners.to_excel(out_path + '.xlsx')  # save time and mbs
        if do_save:
            if pd.notnull(out_path):
                df_winners.to_csv(out_path + '.csv')

        return df_winners


    def find_best_match(self, 
                        left_title,
                        left_title_raw,
                        left_post_merge_id,
                        left_pure_year,
                        right  # can be cut?
                        ):
        # maybe you can vectorize this across the left rows??
        # you should at least be able to stack the pre-processing
        
        #init
        winner = {}
        # no type-matching, legacy dissolver
        right_subset = right
        left_cur_is_journal_type = None
        right_cur_is_journal_type = None       
        
        best_score = -1  # will overwrite immediately as intended
    
        for right_index, right_title in zip(right_subset.index, right_subset.txt):
    
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
        
        return winner

    def improve_merged_table_using_STM_results(self, 
                                               df_total,
                                               chosen_year,
                                               out_path=None,
                                               do_save=False,
                                               cond_len=4,
                                               cond_score=0.6):
        """
        This function looks good
        Careful with the 2 dataframes: they are different

        chosen_year routine is not ideal but OK
        """

        left_raw = df_total[df_total.merge_source == 'pure']
        right_raw = df_total[df_total.merge_source == 'scopus']

        df_core_stm = self.preprocess_and_find_best_matches(left_raw=left_raw,
                                                            right_raw=right_raw,
                                                            out_path=None,
                                                            do_save=False)



        # make minlen
        df_core_stm['minlen'] = df_core_stm.apply(lambda x: np.min([len(x.pure_title_raw.split(' ')), len(x.scopus_title_raw.split(' '))]), axis=1)
        
        # step 1: filter down on the accepted matches
        df_a = df_core_stm[(df_core_stm.minlen > cond_len) & (df_core_stm.score > cond_score)]
        if df_a['pure_title_raw'].value_counts().max() > 1:
            print('there are duplicate raw pure titles: this is unexpected')
        
        # culprit is here: there are some accepted multi right matches...
        # let's apply our fix higher up and then redo
        df_a = (df_a
         .sort_values(['right_post_merge_id','score'],ascending=False)
         .drop_duplicates(subset='right_post_merge_id',keep='first'))
        
        # first define a new ID which is equal left and right
        
        #df_a.left_post_merge_id
        #df_a.right_post_merge_id
        df_a.loc[:,'to_merge_id'] = 1
        df_a.loc[:,'to_merge_id'] = df_a.loc[:,'to_merge_id'].cumsum()
        
        # now merge this information back into df_total
        # first split merged/unmerged to avoid double merges
        df_merged = df_total[df_total.merge_source=='both']
        df_unmerged = df_total[~(df_total.merge_source=='both')]
        df_unmerged_p = df_unmerged[df_unmerged.merge_source=='pure']
        df_unmerged_s = df_unmerged[df_unmerged.merge_source=='scopus']
        #
        # print(len(df_unmerged_p))
        # print(len(df_unmerged_s))
        df_unmerged_p = df_unmerged_p.merge(df_a[['left_post_merge_id','to_merge_id']],
                         left_on='post_merge_id',
                         right_on='left_post_merge_id',
                         how='left')
        df_unmerged_s = df_unmerged_s.merge(df_a[['right_post_merge_id','to_merge_id']],
                         left_on='post_merge_id',
                         right_on='right_post_merge_id',
                         how='left')
        # print(len(df_unmerged_p))
        # print(len(df_unmerged_s))
        #
        # now remove the columns we do not want
        # I am not happy with the hardcode, but OK for now
        
        pure_cols = ['Unnamed: 0',
         'Title of the contribution in original language',
         'Current publication status > Date',
         'Subtitle of the contribution in original language',
         'Type',
         'Workflow > Step',
         'Original language',
         'Electronic version(s) of this work > DOI (Digital Object Identifier)[1]',
         'Organisations > Organisational unit[1]',
         'Organisations > Organisational unit[2]',
         'Organisations > Organisational unit[3]',
         'Organisations > Organisational unit[4]',
         'Organisations > Organisational unit[5]',
         'Organisations > Organisational unit[6]',
         'Organisations > Organisational unit[7]',
         'Organisations > Organisational unit[8]',
         'Organisations > Organisational unit[9]',
         'Organisations > Organisational unit[10]',
         'Journal > Journal[1]:Titles',
         'Journal > Journal[1]:ISSNs',
         'UUID',
         'DOI',
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
         'upw_oa_color_x',
         'upw_oa_color_verbose',
         'ff',
         'ff_provided_organization_string',
         'ff_match',
         'ff_score',
         'ff_terms',
         'ff_message',
         'ff_match_subgroup',
         'DOI_isnull',
         'pub_uuid',
         'pure_year',
         'eid',
         'has_eid',
         'type_contains_book',
         'do_not_merge_on_DOI',
         'is_dupe_based_on_long_title_dupe',
         'do_not_merge_on_title', 
                    'merge_source',
                     'post_merge_id',
                     'to_merge_id']
                            
        scopus_cols = ['Unnamed: 0.1',
         'Unnamed: 0.1.1',
         'abstract_text_clean',
         'aggregationType',
         'corresponding_author_affiliation_id_(matched)',
         'corresponding_author_author_id_(matched)',
         'corresponding_author_indexed_name_(matched)',
         'corresponding_author_surname',
         'corresponding_author_surname_(matched)',
         'deal_discount',
         'deal_discount_verbose',
         'deal_name',
         'deal_owner_raw',
         'deal_scope',
         'diagnostics: ff matching words',
         'diagnostics: ff message',
         'diagnostics: ff raw input ',
         'diagnostics: ff score',
         'diagnostics:corresponding_author_match_score',
         'doi',
         'eIssn',
         'faculty_(matched)',
         'first_VU_author',
         'first_VU_author_raw_organization_info',
         'fulltext_free_url',
         'fund_sponsor',
         'is_a_boai_license',
         'is_a_subscription_journal',
         'is_corresponding_author_a_ukb_author',
         'is_corresponding_author_a_vu_author',
         'is_free_to_read',
         'issn',
         'journal_name',
         'license',
         'open_access_color',
         'subgroup_(matched)',
         'title',
         'upw_oa_color_y',
         'upw_oa_color_category',
         'vu_contact_person',
         'year',
         'scopus_year',
         'upw_oa_color',
         'right_post_merge_id', 
                    'merge_source',
                     'post_merge_id',
                     'to_merge_id',       
         'eid',]
        
        
        # cut cols for safe merging, some columns are double ('merge_source', 'post_merge_id', 'to_merge_id')
        df_unmerged_p = df_unmerged_p[pure_cols]
        df_unmerged_s = df_unmerged_s[scopus_cols]
        
        # construct the merged part
        df_STM_merged = (df_unmerged_p[~(df_unmerged_p.to_merge_id.isnull())]
         .merge(df_unmerged_s[~(df_unmerged_s.to_merge_id.isnull())], 
                left_on='to_merge_id', 
                right_on='to_merge_id', 
                how='inner'))
        df_STM_merged['merge_source'] = 'both'
        df_STM_merged.loc[:,'post_merge_id'] = df_STM_merged.right_post_merge_id  # save back
        df_STM_merged.loc[:,'post_merge_id'].isnull().max()  # should be false
        df_STM_merged = df_STM_merged.drop(columns=['post_merge_id_x', 'post_merge_id_y', 'merge_source_x', 'merge_source_y',
                                    'right_post_merge_id'])  # 'left_post_merge_id',
        
        #
        # construct the unmerged part
        df_STM_unmerged = (df_unmerged_p[(df_unmerged_p.to_merge_id.isnull())]
                           .append(df_unmerged_s[(df_unmerged_s.to_merge_id.isnull())])
                          )
        
        # so now we basically did the following
        # the original stable df is df_total
        # df_total = df_merged + df_unmerged
        # we basically merge using STM within df_unmerged
        # such that we can replace df_unmerged with df_STM_merged + df_STM_unmerged
        #
        # so our final result is df_total_with_STM = df_merged + df_STM_merged + df_STM_unmerged
        
        df_total_with_STM = df_merged.append(df_STM_merged).append(df_STM_unmerged)
        
        # now isolate 2018 on a rich rule and save that as well
        # because we now the data is only proper for the middle years
        
        # ! this is hardcode must be variabilized with year_range[1:-1]
        df_total_with_STM_rich_2018 = df_total_with_STM[(df_total_with_STM.scopus_year == chosen_year) |
                                                        (df_total_with_STM.pure_year == chosen_year)]
        
        # save it
        if do_save:
            df_total_with_STM.to_csv(out_path + '/df_total_with_STM.csv')
            df_total_with_STM_rich_2018.to_csv(out_path + '/df_total_with_STM_2018.csv')

        return df_total_with_STM, df_total_with_STM_rich_2018
    

class faculty_finder:
    """
    This class is a faculty_finder
    It takes in an organizational chart and provides methods to guess faculties from provided organization strings
    These organization strings can be human-written organizational information
    Multiple parts separated by commas are also allowed
    organizational chart: a dataframe with column 'faculteit' containing the faculties,
                          a column 'naam' containing the groups that fall under it, with one row per group.
    """

    # option to add levenschtein or something else to allow correction for typos,

    def __init__(self, organizational_chart, allow_faculty_matching=True, silent=True):
        self.organizational_chart = organizational_chart

        df_n = pd.DataFrame(organizational_chart.faculteit.unique()).rename(columns={0:'fac_orig'})
        df_n['fac_clean'] = df_n['fac_orig'].str.lower().str.strip(' ')
        df_n['fac_alternative'] = df_n['fac_clean'].apply(lambda x: x+'s' if x[-7::] == 'science' else x)
        # improvement idea: add 'humanity' or stem as alternative too or a general alias list
        self.faculty_matrix = df_n

        self.allow_faculty_matching = allow_faculty_matching  # does prevent all faculty-matching

    def match(self, organization_string):
        """
        This function can be improved by using levenschtein to correct for typos
        :param organization_string:
        :return:
        """



        if (organization_string is None) | (pd.isnull(organization_string)):
            score = np.nan  # 0-1
            winner = np.nan
            terms = np.nan
            result_message = 'organization string is empty'
            winner_subgroup = np.nan

        else:
            # init
            df_n = self.faculty_matrix
            score = np.nan  # 0-1
            winner = np.nan
            terms = np.nan
            result_message = 'no faculty name match and bag of words only has trivial words'
            winner_subgroup = np.nan

            # find the interesting part
            # first do a check if faculty is simply in there
            # iterate over parts of the cur_org spit by commas
            # ! This loop overwrites the winner if two comma-split parts both have matches, disregarding match quality
            #
            for cur_org_part in organization_string.split(','):  # split by comma, remove spaces
                # clean spaces, brackets, upper-casing
                cur_org_part_clean = cur_org_part.strip(' ').lower().replace('(', '').replace(')', '')

                if self.allow_faculty_matching & ((cur_org_part_clean in df_n.fac_clean.unique())
                                                  | (cur_org_part_clean in df_n.fac_alternative.unique())):

                    # find the fac_orig that belongs to this and report that
                    winner = df_n.loc[df_n[df_n == cur_org_part_clean].dropna(how='all').index, 'fac_orig'].iloc[0]
                    score = 1  # prevents new entries
                    terms = np.nan
                    result_message = 'matched on faculty name'
                    winner_subgroup = np.nan

                else:
                    # match with bag of words

                    # first clean and prepare the provided organization string argument
                    org_words = nltk.word_tokenize(cur_org_part_clean)
                    remove_words_org = ['vu', 'amsterdam', 'vrije', 'universiteit', 'free', 'university', 'department', 'of', 'the',
                                        'in', 'and', 'a', '@',
                                        'center', 'centre', 'instituut', 'institute', '&', 'for', '(', ')', 'insitute', 'research']
                    org_words_c = [word for word in org_words if word not in remove_words_org]

                    if len(org_words_c) > 0:  # only continue if there are words left
                        # now for the organizational_chart use
                        # loop over all rows of the organizational chart
                        # subgroep = organizational_chart.naam // cur_organizational_chart
                        for cntr, cur_organizational_chart in enumerate(self.organizational_chart.naam.tolist()):
                            # clean the entry under sub-organization
                            cur_orgname = cur_organizational_chart.strip(' ').lower().replace('(', '').replace(')', '')
                            # next tokenize it for bag of words
                            cur_orgname = nltk.word_tokenize(cur_orgname)
                            # remove words which add no distinguishing value (for example, the word 'university')
                            cur_orgname = [word for word in cur_orgname if word not in remove_words_org]
                            # remove commas
                            cur_orgname = [word for word in cur_orgname if word not in [',']]
                            # on purpose, no stemming is done
                            # compute the score through jaccard similarity
                            cur_score = get_jaccard_sim(org_words_c, cur_orgname)
                            # store the highest score across the loop
                            if (np.isnan(score)) | (cur_score > score):
                                score = cur_score
                                winner = self.organizational_chart.loc[cntr,'faculteit']
                                #print([cntr, score, winner, winner_subgroup])
                                #print([winner, cntr])
                                terms = cur_orgname  # comes from organizational_chart
                                result_message = 'matched on bag of words similarity'
                                winner_subgroup = cur_organizational_chart


                            else:
                                try:
                                    if (score == 1) & (cur_score == score) & (len(cur_orgname) >= (len(terms))):
                                        ###print('v1')
                                        # tie-breaker
                                        ###print([cur_orgname, terms])
                                        ###print(len(cur_orgname) >= (len(terms)))
                                        if (org_words_c == cur_orgname):
                                            ###print( org_words_c, cur_orgname , terms)
                                            # if this exact equality holds, then you win and overwrite, otherwise not
                                            score = cur_score
                                            winner = self.organizational_chart.loc[cntr, 'faculteit']
                                            # print([cntr, score, winner, winner_subgroup])
                                            # print([winner, cntr])
                                            terms = cur_orgname  # comes from organizational_chart
                                            result_message = 'matched on bag of words similarity'
                                            winner_subgroup = cur_organizational_chart
                                except:
                                    pass  # do nothing

            if score == 0:
                # this means there were usable words, but no match
                # hence there is no winner, and we can't appoint the top row as winner either
                score = np.nan
                winner = np.nan
                terms = np.nan
                result_message = 'no faculty name match and no bag of words match despite non-trivial words'
                winner_subgroup = np.nan

            # alternatively, we can return zeros as score instead of nans, right now they are interchangeable

        dict_results = {'ff_provided_organization_string': organization_string,
                        'ff_match': winner,
                        'ff_score': score,
                        'ff_terms': terms,
                        'ff_message': result_message,
                        'ff_match_subgroup': winner_subgroup}

        return dict_results

    def match_nan(self):
        return {'ff_provided_organization_string': np.nan,
                'ff_match': np.nan,
                'ff_score': np.nan,
                'ff_terms': np.nan,
                'ff_message': np.nan,
                'ff_match_subgroup': np.nan}

    def match_inexact(self, organization_string):
        """
        [under construction]
        """

def levenshtein(s, t):
    """
    Computes the levenschtein distance between two strings of text
    This is borrowed code and has been checked, but stay careful
    :param s: first string of text
    :param t: second string of text
    :return: a distance measure, not normalized
    """
    if s == "":
        return len(t)
    if t == "":
        return len(s)
    if s[-1] == t[-1]:
        cost = 0
    else:
        cost = 1
    res = min([levenshtein(s[:-1], t) + 1,
               levenshtein(s, t[:-1]) + 1,
               levenshtein(s[:-1], t[:-1]) + cost])
    return res



def get_dist(str1, str2):
    """
    Computes the normalized levenschtein distance between two strings of text
    :param s: first string of text
    :param t: second string of text
    :return: a distance measure, normalized
    """
    maxlen = np.max([len(str1), len(str2)])
    return levenshtein(str1, str2) / maxlen  # normalize by len of longest string, 0 = perfect match, 1 = no match


def get_dist_fast(str1,str2):
    """
    A faster Cython implementation for levenshtein. Use 'pip install editdistance' to make it work.
    :param str1: first string
    :param str2: second string
    :return: levenshtein or edit distance, 0 is closest and best match, 1 is furthest

    Copyright information for this specific function editdistance.eval() used in this get_dist_fast():
    edit distance
    Copyright (c) 2013 Hiroyuki Tanaka
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documen
    tation files (the "Software"), to deal in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
    persons to whom the Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of
    the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
     WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS
     OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
      OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
    """
    return editdistance.eval(str1, str2) / np.max([len(str1), len(str2)])


def get_jaccard_sim(str1, str2):
    """
    Computes the jaccard similarity value
    :param str1: first string of text
    :param str2: second string of text
    :return: jaccard similarity value between the two
    example: get_jaccard_sim(['clino','developmental','a'],['clinical', 'neuro-', 'developmental', 'psychology'])
    """
    a = set(str1)
    b = set(str2)
    c = a.intersection(b)
    return float(len(c)) / float(len(a) + len(b) - len(c))



class corresponding_author_functions():
    def __init__(self):
        pass

    @staticmethod
    def match_aut(row):

        # because surnames are short (usually <10~20 characters) and a few words, we can use levenshtein to get some scoring
        # Assumption 1: the corresponding author always appears in the authorgoup list if both fields are not empty
        #               such that one of them is the corresponding author
        # Assumption 2: the corresponding author is the author with the highest levenshtein score
        # we need to check this, but for now it seems like a reasonable approximation and we can start getting numbers

        # speed-up: idea: corr_author is prolly in front 3, so we should build a zero-stoppe
        # update: it sped it up 1000x or so
        # print('beep')
        if (row.has_correspondence_surname) and (row.has_authorgroup_surnames):
            left_aut = row['corresponding_author_surname']
            lowest_res = np.Inf
            best_author = np.nan
            multiple_winners = False

            # print(left_aut)

            df_aut = pd.DataFrame(row.abstract_object.authorgroup)[['affiliation_id', 'surname', 'indexed_name',
                                                                    'auid']]
            do_try_new_aut = True

            right_aut_auid_list = list(df_aut.auid.unique())
            right_aut_auid_list = [x for x in right_aut_auid_list if x is not None]  # remove auids of None
            for right_aut_auid in right_aut_auid_list:  # ASSUME UNIQUE, we will use this as a key here: LATER: TO .APPLY

                # print(1)
                # print(right_aut_auid)

                if do_try_new_aut:
                    # print('in')
                    right_aut_rows = df_aut[df_aut.auid == right_aut_auid]  # .surname.iloc[0]
                    right_aut = df_aut[df_aut.auid == right_aut_auid].surname.iloc[0]
                    # print('in2')
                    # print(left_aut,right_aut)
                    res = get_dist_fast(left_aut, right_aut)  # get_dist_fast ipv get_dist
                    # print('in3')
                    if res < lowest_res:
                        lowest_res = res
                        best_aut = right_aut
                        best_aut_rows = right_aut_rows
                        out = list(best_aut_rows[['affiliation_id', 'surname', 'indexed_name',
                                                  'auid']].iloc[0])

                        winner_surname = best_aut_rows[['affiliation_id', 'surname', 'indexed_name',
                                                  'auid']].iloc[0].surname
                        out.append(lowest_res)
                    elif res == lowest_res:
                        ######print('equal levenshtein value, keeping previous')  ### for later testing
                        pass

                    if res == 0:
                        do_try_new_aut = False


                    my_return = out

                    # print('res is ' + str(res))

                else:
                    pass

                # print(2)
            df_mw = pd.DataFrame(row.abstract_object.authorgroup)
            multiple_exact_matches = len(df_mw[df_mw.surname == winner_surname]) > 1
            my_return.append(multiple_exact_matches)  # not out.append here
        else:
            my_return = [None, None, None,
                            None, None,
                            None] #np.nan

        return my_return

    @staticmethod
    def check_if_corresponding_author_is_chosen_affil_author(row, afids):
        # is the matched corresponding author (to get afid) a vu/ukb/we-author?
        try:
            if not (set(row.match_affiliation_id.split(', ')).isdisjoint(set(afids))):
                res = True
            else:
                res = False
        except:
            res = np.nan
        return res


    def add_corresponding_author_info(self,df,vu_afids,ukb_afids):
        # df must have column abstract_object with in it a correspondence field (may require re-running abstract yourself)

        try:
            df.abstract_object
        except:
            print('dataframe has no column named abstract_object')

        try:
            df.abstract_object.iloc[0].correspondence
        except:
            print('dataframe abstract object has no field named correspondence. This only checks first row.')

        # debugging results
        df['has_abstract_info'] = ~df.abstract_object.isnull()

        def check_if_it_has_correspondence(row):
            if row.has_abstract_info == False:
                result = False
            elif row.abstract_object.correspondence is None:  # does not accoun for nans
                result = False
            else:
                result = True
            return result

        def check_if_it_has_authorgroup(row):
            if row.has_abstract_info == False:
                result = False
            elif row.abstract_object.authorgroup is None:  # does not accoun for nans
                result = False
            else:
                result = True
            return result

        # do we have correspondence info
        df['has_correspondence'] = df.apply(check_if_it_has_correspondence, axis=1)

        # what is the surname?
        df['corresponding_author_surname'] = df.apply(
            lambda x: np.nan if x.has_correspondence is False else x.abstract_object.correspondence.surname,
            axis=1)
        df['has_correspondence_surname'] = ~df.corresponding_author_surname.isnull()
        df.at[df[df.issn.apply(lambda x: True if isinstance(x, list) else False)].index.tolist(), 'issn'] = None
        # this part is senstive: please make an automated test here or rewrite it thoroughly !
        df['has_authorgroup'] = df.apply(check_if_it_has_authorgroup, axis=1)
        df['has_authorgroup_surnames'] = ~df.apply(
            lambda x: np.nan if x.has_authorgroup is False else x.abstract_object.authorgroup[0][9], axis=1).isnull()

        df[['match_affiliation_id', 'match_surname', 'match_indexed_name',
            'match_auid', 'match_aut_score',
            'has_multiple_exact_matches']] = df.apply(
            self.match_aut, axis=1, result_type="expand")

        df['is_corresponding_author_a_vu_author'] = df.apply(self.check_if_corresponding_author_is_chosen_affil_author,
                                                             afids=vu_afids, axis=1)
        df['is_corresponding_author_a_ukb_author'] = df.apply(self.check_if_corresponding_author_is_chosen_affil_author,
                                                              afids=ukb_afids, axis=1)

        return df


def stack_titles(row):
    notnulls = 0
    if pd.notnull(row['Subtitle of the contribution in original language']):
        part2 = row['Subtitle of the contribution in original language']
        part2_null = False
    else:
        part2 = ''
        part2_null = True
    if pd.notnull(row['Title of the contribution in original language']):
        part1 = row['Title of the contribution in original language']
        part1_null = False
    else:
        part1 = ''
        part1_null = True

    if part1_null | part2_null :
        return part1 + part2
    else:
        return part1 + ': ' + part2


# let's prepare the preprocessing
def remove_punctuation(text):
    '''a function for removing punctuation'''
    import string
    # replacing the punctuations with no space,
    # which in effect deletes the punctuation marks
    translator = str.maketrans('', '', string.punctuation)
    # return the text stripped of punctuation marks
    return text.translate(translator)


def get_abstract_if_any(row):
    # always run comma_space_fix first
    try:
        if row.abstract_object.abstract is None:
            return ' '
        else:
            return row.abstract_object.abstract
    except:
        return ' '


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


def remove_numbers(text):
    translator = str.maketrans('', '', '0123456789')
    return text.translate(translator)

def remove_stopwords_and_lower(text):
    '''a function for removing the stopword'''
    # extracting the stopwords from nltk library
    sw = stopwords.words('english')
    # displaying the stopwords
    np.array(sw)
    # we know more stuff no one needs at all like 'department' but let's keep them for now
    # removing the stop words and lowercasing the selected words
    text = [word.lower() for word in text.split() if word.lower() not in sw]
    # joining the list of words with space separator
    return " ".join(text)













