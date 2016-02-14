import numpy as np
import matplotlib as plt
import os
import pandas as pd 

filename = os.path.join('airline-twitter-sentiment','Tweets.csv')
tweets = pd.read_csv(filename)


tw_text = tweets.text
tw_name = tweets.name
tw_sent = tweets.airline_sentiment
tw_sent_conf = tweets.airline_sentiment_confidence
tw_air = tweets.airline


print tw_text[16]
