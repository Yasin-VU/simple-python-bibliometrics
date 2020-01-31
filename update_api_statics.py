# running this function will update the pickled static response objects
# we use these for shortcuts like mimicking unpaywall response on invalid DOIs and mocks
# wacht the path

from static import PATH_START, PATH_START_PERSONAL
from static import PATH_START_SERVER , PATH_START_PERSONAL_SERVER
import requests
import pickle  # will cPickle under the hood
from static import PATH_STATIC_RESPONSES
from static import PATH_STATIC_RESPONSES_ALTMETRIC

do_altmetric = True

if do_altmetric:
    # altmetric
    #api_ver = 'v1'  # may change in future, so here it is. For api-key re-edit with altmetric package
    #api_url = "http://api.altmetric.com/%s/" % api_ver
    url = 'https://api.altmetric.com/v1/doi/10.1038/480426ax'
    r = requests.get(url, params={}, headers={})
    r.connection.close()
    file_name = PATH_STATIC_RESPONSES_ALTMETRIC
else:
    # unpaywall
    r = requests.get("https://api.unpaywall.org/invalid?email=2@2.nl")  # get response on invalid request
    r.connection.close()  # this removes the link to the connection pool in order to allow pickling
    file_name = PATH_STATIC_RESPONSES

print('-----')
print(r.json())
msg = 'check if response above is what you want, enter a y character to continue overwriting else press any other character'
txt = input(msg)
if txt == 'y':
    out_file = open(file_name, 'wb')
    pickle.dump(r, out_file)
    out_file.close()
    print('updated file')
else:
    print('not saving anything')
print('done')

#in_file = open(file_name, 'rb')
#data = pickle.load(in_file)
#in_file.close()
#print(data)
#print(data.json())

