# Welcome to examples.py with examples to get started with this package
# The goal of this package is to provide you user-friendly commands to download and prepare bibliometrics data
# most lines are comments, with every example on average just 3 lines of code
# Let's get started

# example 0: your first dataset
# it is known bibliometrics data exists and is out there, but it needs to get on the computer
# preferably without any manual actions like visiting websites and pressing download buttons
# for this purpose, one can use API's (Application Programmable Interfaces) and download data directly without leaving Python (hurray)
# below a first dataset using Scopus
#
# first one must define a search query to tell Scopus which data is wanted (the entire corpus is simply too large)
# this query asks Scopus to return records of publications with 'Gunes' in the author names and 'tensor' in the title.
my_query = "AUTH(Gunes) AND TITLE(TENSOR)"
#
# Next this query must be sent to Scopus, through the API
# There are two ways to do this: either by using general API commands or by using a 'wrapper'.
# The 'wrapper' is a package of functions which simplifies the general API commands
# One wrapper for scopus search is the following
from pybliometrics.scopus import ScopusSearch
# knowledge of python packages is assumed
# if that command failed, install the package pybliometrics
# during your first run an api-key will be asked, get one from scopus
# hint: in production code always put all imports at the top
#
# now send out the query
# easy querying of Scopus
s = ScopusSearch(my_query)
# the variable s stores the results
#
# next we turn it into a pandas dataframe for easy handling
# we use s.results to make the wrapper return the results in suitable format
# and pandas for data handling
import pandas as pd
df = pd.DataFrame(s.results)
#
# now the data can be printed
print(df.head())
#
# this wraps up how to get scopus dsta through python automatically


# example 1: enriching your data with Altmetric
# scopus is not everyhing
# for example, it has no twitter mentions counts or not of sufficient quality
# but a different API sercice named Altmetric does
# so Altmetric can be used to 'enrich' Scopus records
# then both information sources are combined
#
# Using our package, this is as simple as a single command
from core_functions import add_altmetric_columns  # will be renamed later
df = add_altmetric_columns(df)
#
# and that is all, now the dataframe df has altmetric information for every
# record
# the only requirement is that df has a column 'doi' and no overlapping columns
# with the altmetric columns
#
# to summarize: only 1 function and 1 argument is needed to get Altmetric data

# example 2: enriching your data with Unpaywall
# Unpaywall is an API service for open access status of publications
# tells if papers are free to read and what its url is amongst other things
#
# and again just a single function, import and call, done
from core_functions import add_unpaywall_columns
df = add_unpaywall_columns(df)
#
# these 2 lines enrich any dataframe with a doi column with unpaywall
# furthermore, the function is optimized and will run 40x faster than usual
# basically, the simple-python-bibliometrics package harvests unpaywall
# roughly 40x faster than a basic code would do


# more examples are underway
# like scopus abstracts, author info processing, merging scopus with local
# university/library repositories, and much more




























