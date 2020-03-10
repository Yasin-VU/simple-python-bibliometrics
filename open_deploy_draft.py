"""
Code to process a dataframe into X_test for ML deployment
"""

# import packages
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, KFold
from nltk.stem.snowball import SnowballStemmer
from scipy.stats import randint
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
# from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.feature_selection import chi2
from nltk.corpus import stopwords
#
# if nltk does not work, then run the 3 lines below:
### import nltk
### nltk.download("punkt")
### nltk.download("stopwords")

def predict_faculty(df_in, model, tfidf):
    """
    :param df: conditions exist
    :return:
    """
    X = get_X_and_y(df_in)
    features_tfidf = tfidf.transform(X['text_info']).toarray()  # don't push all feats in, just the text stack
    features = pd.concat([pd.DataFrame(features_tfidf), X.reset_index().drop(columns={'text_info',
                                                                                      'text_info_read',
                                                                                      'text_info_export',
                                                                                      'index'})], axis=1)
    y_pred = model.predict(features)

    return y_pred




# Here are copies of common functions for the sole purpose of easing imports for cross-platform use
#
# let's prepare the preprocessing
def remove_punctuation(text):
    '''a function for removing punctuation'''
    import string
    # replacing the punctuations with no space,
    # which in effect deletes the punctuation marks
    translator = str.maketrans('', '', string.punctuation)
    # return the text stripped of punctuation marks
    return text.translate(translator)


# print(remove_punctuation("""123, hi, ./;[][90-] \][0-*( )] hi man how are you""" ))  # powerful and fast

def remove_numbers(text):
    import string
    translator = str.maketrans('', '', '0123456789')
    return text.translate(translator)


# print(remove_numbers('8u1981723 asdh 288 hi hi 2 hi '))  # nice


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


def comma_space_fix(text):
    return (text
            .replace(": ", ":")
            .replace(":", ": ")
            .replace("! ", "!")
            .replace("!", "! ")
            .replace("? ", "?")
            .replace("?", "? ")
            .replace(", ", ",")
            .replace(",", ", ")
            .replace(". ", ".")
            .replace(".", ". ")
            .replace("; ", ";")
            .replace(";", "; "))  # this makes both ",x" and ", x" ", x"


# hey! notice that you are cutting off the org info after the first comma but lumping it all together now
# for multi-affil, this may not be what you want as it loses ordering
# however it is OK for now

def remove_common_words_and_lower(text):
    # we need to remove vu/dept/amsterdam because it messes up the bigrams
    remove_words_org = ['vu', 'amsterdam', 'vrije', 'universiteit', 'free', 'university', 'department', 'of', 'the',
                        'in',
                        'and', 'a', '@', 'center', 'centre', 'instituut', 'institute', '&', 'for', '(', ')', 'insitute',
                        'research']
    #
    # removing institute is perhaps not the best option, try stuff out : )
    # removing the stop words and lowercasing the selected words
    text = [word.lower() for word in text.split() if word.lower() not in remove_words_org]
    # joining the list of words with space separator
    return " ".join(text)


# fill empty nans with empty strings,
# this difference avoids errors in type assertions
def fill_empty(row):
    if pd.notnull(row):
        return row
    else:
        return ''

# define encoding/enumeration
def encode_fac(row):
    if row == 'Faculty of Science':
        id = 0
    elif row == 'Faculty of Behavioural and Movement Sciences':
        id = 1
    elif row == 'medical center':
        id = 2
    elif row == 'Faculty of Social Sciences':
        id = 3
    elif row == 'School of Business and Economics':
        id = 4
    elif row == 'Faculty of Law':
        id = 5
    elif row == 'Faculty of Humanities':
        id = 6
    elif row == 'Faculty of Religion and Theology':
        id = 7
    elif row == 'ACTA':
        id = 8
    else:  # rest
        id = 9
    return id


def get_X_and_y(df):

    def add_space(row):
        return row + ' '

    df['text_info_1'] = (df
                         .first_VU_author_raw_organization_info
                         .apply(fill_empty)
                         .apply(comma_space_fix)
                         .apply(remove_punctuation)
                         .apply(remove_numbers)
                         .apply(remove_stopwords_and_lower)
                         .apply(remove_common_words_and_lower)
                         .apply(add_space))
    df['text_info_2'] = (df
                         .title
                         .apply(fill_empty)
                         .apply(comma_space_fix)
                         .apply(remove_punctuation)
                         .apply(remove_numbers)
                         .apply(remove_stopwords_and_lower)
                         .apply(remove_common_words_and_lower)
                         .apply(add_space))
    df['text_info_3'] = (df
                         .journal_name
                         .apply(fill_empty)
                         .apply(comma_space_fix)
                         .apply(remove_punctuation)
                         .apply(remove_numbers)
                         .apply(remove_stopwords_and_lower)
                         .apply(remove_common_words_and_lower)
                         .apply(add_space))

    df['text_info_4'] = (df
                         .abstract_text_clean
                         .apply(fill_empty)
                         .apply(comma_space_fix)
                         .apply(remove_punctuation)
                         .apply(remove_numbers)
                         .apply(remove_stopwords_and_lower)
                         .apply(remove_common_words_and_lower)
                         .apply(add_space))
    # define the features matrix
    # notice that for this setting we do not add extra cols
    # for example, we could add #authors as a column
    # and let the machine learning decide how/if to use that

    abstract_down_weight = 3  # hinges on space_fix
    #
    df['text_info'] = (3 * df['text_info_1']
                       + ' '
                       + 3 * df['text_info_2']  # title
                       + ' '
                       + 3 * df['text_info_3']  # journal_name
                       + ' '
                       + df['text_info_4'])  # abstract

    df['text_info_read'] = (df['text_info_1']
                            + ' || '
                            + df['text_info_2']
                            + ' || '
                            + df['text_info_3']
                            + ' || '
                            + df['text_info_4']
                            )

    df['text_info_export'] = (
            ' #ORGVU1 ' +
            df['text_info_1']
            + ' #TITLE '
            + df['text_info_2']
            + ' #JNAME '
            + df['text_info_3']
            + ' #ABS '
            + df['text_info_4']
    )

    for id in np.arange(0, 10):
        df['fac_' + str(id)] = df['faculty_(matched)'].apply(encode_fac) == id

    # extra feature
    df['has_DOI'] = df.apply(lambda x: True if pd.notnull(x.DOI) else False, axis=1)


    X = df[
        ['text_info', 'text_info_read', 'text_info_export', 'has_DOI', 'type_contains_book', 'fac_0', 'fac_1',
         'fac_2', 'fac_3', 'fac_4', 'fac_5'
            , 'fac_6', 'fac_7', 'fac_8', 'fac_9']]

    # add contact to train AR on later and then remove it

    return X
