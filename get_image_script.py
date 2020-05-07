from requests import exceptions
import pandas as pd
import argparse
import requests
import cv2
import os

def get(term, OUTPUT='./original'):
    API_KEY = '******************'
    MAX_RESULTS = 255
    MAX_SIZE = 255
    GROUP_SIZE = 50

    # 取得したエンドポイントURL
    URL = 'https://api.cognitive.microsoft.com/bing/v7.0/images/search'

    if not os.path.isdir(OUTPUT):
        os.mkdir(OUTPUT)

    EXCEPTIONS = set([IOError, FileNotFoundError,
        exceptions.RequestException, exceptions.HTTPError,
        exceptions.ConnectionError, exceptions.Timeout])

    headers = {'Ocp-Apim-Subscription-Key' : API_KEY}
    params = {
        'q': term,
        'offset': 0,
        'count': GROUP_SIZE,
        'imageType':'Photo',
        'color':'ColorOnly'
    }

    # make the search
    print('[INFO] searching Bing API for "{}"'.format(term))
    search = requests.get(URL, headers=headers, params=params)
    search.raise_for_status()

    # grab the results from the search, including the total number of
    # estimated results returned by the Bing API
    results = search.json()
    estNumResults = min(results['totalEstimatedMatches'], MAX_RESULTS)
    print('[INFO] {} total results for "{}"'.format(estNumResults, term))

    # initialize the total number of images downloaded thus far
    total = 0

    # loop over the estimated number of results in `GROUP_SIZE` groups
    for offset in range(0, estNumResults, GROUP_SIZE):
        # update the search parameters using the current offset, then
        # make the request to fetch the results
        print('[INFO] making request for group {}-{} of {}...'.format(offset, offset + GROUP_SIZE, estNumResults))
        params['offset'] = offset
        search = requests.get(URL, headers=headers, params=params)
        search.raise_for_status()
        results = search.json()
        print('[INFO] saving images for group {}-{} of {}...'.format(offset, offset + GROUP_SIZE, estNumResults))
        # loop over the results
        for v in results['value']:
            # try to download the image
            try:
                # make a request to download the image
                print('[INFO] fetching: {}'.format(v['contentUrl']))
                r = requests.get(v['contentUrl'], timeout=30)

                # build the path to the output image
                ext = v['contentUrl'][v['contentUrl'].rfind('.'):v['contentUrl'].rfind('?') if v['contentUrl'].rfind('?') > 0 else None]
                p = os.path.sep.join([OUTPUT, '{}{}'.format(str(total).zfill(8), ext)])

                # write the image to disk
                f = open(p, 'wb')
                f.write(r.content)
                f.close()

            # catch any errors that would not unable us to download the
            # image
            except Exception as e:
                # check to see if our exception is in our list of
                # exceptions to check for
                if type(e) in EXCEPTIONS:
                    print('[INFO] skipping: {}'.format(v['contentUrl']))
                    continue
            # try to load the image from disk
            image = cv2.imread(p)

            # if the image is `None` then we could not properly load the
            # image from disk (so it should be ignored)
            if image is None:
                print('[INFO] deleting: {}'.format(p))
                os.remove(p)
                continue

            # update the counter
            total += 1
            if total >= 255:
                return

def rename(dir):
    file_list = os.listdir(dir)
    for _ in file_list:
        name = _.split('.')
        if len(name) == 1 or name[1] == 'L':
            try:
                os.rename(dir+_, dir+name[0]+'.png')
            except:
                os.remove(dir+_)

target_csv = './original/target.csv'
target = pd.read_csv(target_csv,)

for _ in target.values:
    print(_)
    term = _[0]
    dir = './original/{0}/'.format(_[1])
    if _[2] == 1.0:
        continue
    try:
        get(term, dir)
        rename(dir)
        target.loc[target['name'] == _[0], 'ignore_flag'] = 1
        target.to_csv(target_csv, index=False)
    except:
        continue
