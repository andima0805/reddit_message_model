#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 19:52:15 2020
@author: andima
"""

import praw
import pandas as pd
import pytz
from my_functions import *
from datetime import datetime
import pickle
import sys
from sklearn.decomposition import PCA

subreddit_channel = 'politics'

reddit = praw.Reddit(
     client_id="xxx",
     client_secret="xxx",
     user_agent="testscript by u/fakebot3",
     username="xxx",
     password="xxx!",
     check_for_async=False
 )

print(reddit.read_only)

def conv_time(var):
    tmp_df = pd.DataFrame()
    tmp_df = tmp_df.append(
        {'created_at': var},ignore_index=True)
    tmp_df.created_at = pd.to_datetime(
        tmp_df.created_at, unit='s').dt.tz_localize(
            'utc').dt.tz_convert('US/Eastern') 
    return datetime.fromtimestamp(var).astimezone(pytz.utc)

def get_reddit_data(var_in):
    import pandas as pd
    tmp_dict = pd.DataFrame()
    tmp_time = None
    try:
        tmp_dict = tmp_dict.append({"created_at": conv_time(
                                        var_in.created_utc)},
                                    ignore_index=True)
        tmp_time = tmp_dict.created_at[0] 
    except:
        print ("ERROR")
        pass
    tmp_dict = {'msg_id': str(var_in.id),
                'author': str(var_in.author),
                'body': var_in.body, 'datetime': tmp_time}
    return tmp_dict

def pre_process_classify_stem(text_in):
    import re
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    import enchant
    aaa = enchant.Dict("en_US")
    bbb = PorterStemmer()
    sw = stopwords.words('english')
    text_in = re.sub('[^A-z]+', " ", text_in).lower().split()
    text_in = [word for word in text_in if word not in sw]
    text_in = [bbb.stem(word) for word in text_in]
    text_in = [word for word in text_in if aaa.check(word)]
    text_in = ' '.join(text_in)
    return text_in

for comment in reddit.subreddit(subreddit_channel).stream.comments():
    tmp_df = get_reddit_data(comment)
    body=tmp_df["body"]

    # Cleaning text
    cleaned = clean_text(body)

    # Remove stopwords
    sw_2 = rem_sw(cleaned)

    # stemming
    stem_2 = pre_process_classify_stem(sw_2)

    # vectorizer
    with open("vectorizer.pkl", 'rb') as pickle_file:
        vec_2 = pickle.load(pickle_file)

    my_vec_t = pd.DataFrame(vec_2.transform([stem_2]).toarray())
    my_vec_t.columns = vec_2.vocabulary_

    # PCA
    with open("pca.pkl", 'rb') as pickle_file:
        pca_2 = pickle.load(pickle_file)

    my_pca_vec_2 = pca_2.transform(my_vec_t)

    # model
    with open("my_model.pkl", 'rb') as pickle_file:
        my_model_2 = pickle.load(pickle_file)

    pred = my_model_2.predict(my_pca_vec_2)
    the_preds = pd.DataFrame(my_model_2.predict_proba(my_pca_vec_2))
    the_preds.columns = my_model_2.classes_

    probs = the_preds.loc[0, :].values.tolist()
    print(the_preds)

    if probs[0]>probs[1]:
        print("The message belongs to ", my_model_2.classes_[0])

    elif probs[0]< probs[1]:
        print("The message belongs to ", my_model_2.classes_[1])

    else:
        print("The message is neutral")
    



