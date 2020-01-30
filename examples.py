# Welcome to examples.py with examples to get started with this package
# The goal of this package is to provide you user-friendly commands to download and prepare bibliometrics data
# Let's get started

# example 0: your first dataset
# it is known bibliometrics data exists and is out there, but it needs to get on the computer
# preferably without any manual actions like visiting websites and pressing download buttons
# for this purpose, one can use API's (Application Programmable Interfaces) and download data directly without leaving Python (hurray)
# below a first dataset using Scopus
#
# first one must define a search query to tell Scopus which data is wanted (the entire corpus is simply too large)
# this query asks Scopus to return records of publications with 'Gunes' in the author names and 'tensor' in the title.
my_query = "AUTHOR(Gunes) AND TITLE(TENSOR)"  
# 
# Next this query must be sent to Scopus, through the API
# There are two ways to do this: either by using general API commands or by using a 'wrapper'.
# The 'wrapper' is a package of functions which simplifies the general API commands
# One wrapper for scopus search is the following
from pybliometrics.scopus import ScopusSearch
# knowledge of python packages is assumed
# if that command failed, install the package pybliometrics
# during your first run an api-key will be asked, get one from scopus
#
# now send out the query
# easy querying of Scopus
s = ScopusSearch(my_query)
# the variable s stores the res
ults
#
# next we turn it into a pandas dataframe for easy handling
# we use s.results to make the wrapper return the results in suitable format
df = pd.DataFrame(s.results)
#
# now the data can be printed
print(df.head())
#
# this wraps up how to get scopus dsta through python automatically


# example 1: enriching your data with Altmetric

