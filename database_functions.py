# Goal: provide database functions for plug-and-play database use instead of flat files

from import_framework.nlp_functions import faculty_finder
from import_framework.static import UNPAYWALL_EMAIL
import pandas as pd
import calendar
import numpy as np
import requests
from pybliometrics.scopus import ScopusSearch
from pybliometrics.scopus import AbstractRetrieval
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
import time
import datetime
import re
import mysql.connector
from mysql.connector import Error
from mysql.connector import errorcode  # new
from datetime import datetime  # new
import calendar  # new

# imports from our own import framework
#####import sys
#####sys.path.insert(0, 'C:/Users/yasin/Desktop/git/common_functions')  # not needed sometimes
from import_framework.nlp_functions import faculty_finder
from import_framework.nlp_functions import get_dist
from import_framework.nlp_functions import corresponding_author_functions
#
from import_framework.core_functions import add_year_and_month
from import_framework.core_functions import get_scopus_abstract_info
from import_framework.core_functions import get_first_chosen_affiliation_author  # ! i want all vu authors now : )
from import_framework.core_functions import add_unpaywall_columns
from import_framework.core_functions import my_timestamp
from import_framework.core_functions import add_deal_info
from import_framework.core_functions import make_types_native_basic

# not sure if used at all, test it
#
class NumpyMySQLConverter(mysql.connector.conversion.MySQLConverter):
    """ A mysql.connector Converter that handles Numpy types """

    def _float32_to_mysql(self, value):
        return float(value)

    def _float64_to_mysql(self, value):
        return float(value)

    def _int32_to_mysql(self, value):
        return int(value)

    def _int64_to_mysql(self, value):
        return int(value)


def pre_process_for_push(df_to_upload, primary_key_start):
    # timestamp
    ### df_to_upload['Unnamed: 0'] = formatted_date
    #
    # id
    df_to_upload = df_to_upload.reset_index().rename(columns={'index': 'id'})
    df_to_upload['id'] = df_to_upload.id + primary_key_start + 1  # ! primary key
    #
    # cut columns
    df_to_upload = df_to_upload[['id',
                                 'aggregationType',
                                 'creator',
                                 'doi',
                                 'ff_match',
                                 'ff_match_subgroup',
                                 'first_affil_author',
                                 'fund_sponsor',
                                 'publicationName',
                                 'upw_oa_color_category',
                                 'deal_name',
                                 'deal_discount_verbose',
                                 'upw_oa_color_verbose',
                                 'deal_owner_verbose']]
    # edit types (will edit again later)
    df_to_upload.id = df_to_upload.id.astype('int')
    df_to_upload = df_to_upload.fillna('nan')  # ?
    return df_to_upload


def process_df_to_list_to_push(df_to_upload):
    lst_to_push = []
    for ii in np.arange(0, len(df_to_upload)):
        lst_to_push.append(tuple(make_types_native_basic(df_to_upload.iloc[ii].to_list())))
    return lst_to_push


def get_connection(host, database, user, pw):
    connection = mysql.connector.connect(host=host,
                                         database=database,
                                         user=user,
                                         password=pw)
    return connection


def push_df_to_db(connection, df_to_upload):
    # only works for test-situation
    # care: id increment is dealt with elsewhere (not nice)

    try:
        connection.set_converter_class(NumpyMySQLConverter)  # avoid write issue with dtype  ### try without please
        cursor = connection.cursor()

        # try upload id too...
        #
        mySql_insert_query = """INSERT INTO oadata (id, 
                                aggregationType, 
                                creator,
                                doi, 
                                ff_match, 
                                ff_match_subgroup, 
                                first_affil_author, 
                                fund_sponsor, 
                                publicationName, 
                                upw_oa_color_category, 
                                deal_name, 
                                deal_discount_verbose, 
                                upw_oa_color_verbose, 
                                deal_owner_verbose)
                               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) """

        records_to_insert = process_df_to_list_to_push(df_to_upload)
        cursor.executemany(mySql_insert_query, records_to_insert)
        connection.commit()
        print(cursor.rowcount, "Record inserted successfully into table")

    except mysql.connector.Error as error:
        print("Failed to insert record into MySQL table {}".format(error))

    finally:
        if (connection.is_connected()):
            cursor.close()
            connection.close()
            print("MySQL connection is closed")


def run_query(connection, query):
    # no idea if this works for all reads

    silent = False
    try:
        sql_select_Query = query  # "select * from oadata"
        cursor = connection.cursor()
        cursor.execute(sql_select_Query)
        records = cursor.fetchall()
        df_t = pd.DataFrame(records)  # .rename(columns={0: 'eid', 1: 'pub_uuid'})
        if silent is False:
            print("Total number of rows is: ", cursor.rowcount)
        success = True
    except Error as e:
        # always print this, later also add logging
        print("Error reading data from MySQL table", e)
        print('returning None')
        df_t = None
        success = False
    finally:
        if (connection.is_connected()):
            connection.close()
            cursor.close()
            if silent is False:
                print("MySQL connection is closed")
    return df_t, success


def get_last_primary_key(connection):
    df_pull, success = run_query(connection, "SELECT max(id) FROM mydb.oadata")
    last_primary_key = None
    if success:
        last_primary_key = df_pull.iloc[0, 0]
    return last_primary_key, success
