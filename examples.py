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


# below = under construction

import ScopusSearch
# and it allows really easy querying of Scopus
s = ScopusSearch(my_query, refresh=True)


# download package
# imports?
# setup config
# api keys

df = pd.DataFrame(s.results)



# example 1
