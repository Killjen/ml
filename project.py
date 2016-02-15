#!/usr/bin/python2.7
import numpy as np
import matplotlib as mplt
import matplotlib.pyplot as plt
import os
import pandas as pd
import math
from wordcloud import WordCloud

filename = os.path.join('airline-twitter-sentiment','Tweets.csv')
tweets = pd.read_csv(filename)


tw_text = tweets.text
tw_name = tweets.name
tw_sent = tweets.airline_sentiment
tw_sent_conf = tweets.airline_sentiment_confidence
tw_air = tweets.airline
tw_rt = tweets.retweet_count
tw_coord = tweets.tweet_coord
tw_time = tweets.tweet_created
tw_loc = tweets.tweet_location

print len(tw_text)
assert len(tw_text) == len(tw_name) and len(tw_name) == len(tw_sent) and len(tw_name)==len(tw_sent_conf) and len(tw_name)==len(tw_air) and len(tw_name) == len(tw_rt) and len(tw_name)==len(tw_coord) and len(tw_name) == len(tw_time) and len(tw_name)==len(tw_loc)
n=len(tw_loc)

#stats
coord = 0 #available coords
loc = 0 #available locs
rt = 0 #rt > 0
conf = 0 #sent_conf != 1
pos = 0 #sent is positive
neg = 0
neu = 0
for i in xrange(n):
	if type(tw_coord[i]) is not str:
		coord += 1
	if type(tw_loc[i]) is not str:
		loc += 1
	if tw_rt[i] > 0:
		rt += 1
	if tw_sent_conf[i] < 1:
		conf +=1
	if tw_sent[i] == "positive":
		pos += 1
	if tw_sent[i] == "neutral":
		neu += 1
	if tw_sent[i] == "negative":
		neg += 1

print "available coords: " + str(coord) + " out of " + str(n) + " (" + str(100*coord/(1.0*n)) + "%)"
print "available locs: " + str(loc) + " out of " + str(n) + " (" + str(100*loc/(1.0*n)) + "%)"
print "#RT > 0: " + str(rt) + " out of " + str(n) + " (" + str(100*rt/(1.0*n)) + "%)"
print "conf < 1: " + str(conf) + " out of " + str(n) + " (" + str(100*conf/(1.0*n)) + "%)"
print "sentiment positive: " + str(pos) + " out of " + str(n) + " (" + str(100*pos/(1.0*n)) + "%)"
print "sentiment neutral: " + str(neu) + " out of " + str(n) + " (" + str(100*neu/(1.0*n)) + "%)"
print "sentiment negative: " + str(neg) + " out of " + str(n) + " (" + str(100*neg/(1.0*n)) + "%)"

#generate WordCloud
#see https://github.com/amueller/word_cloud

#generate one string of text
text = ' '.join(tw_text)

# take relative word frequencies into account, lower max_font_size
def wcloud(text, filename):
	wordcloud = WordCloud(width=1000, height=500).generate(text) #, max_font_size=85, relative_scaling=.5
	plt.figure(figsize=(10,5))
	plt.imshow(wordcloud)
	plt.axis("off")
	fig = plt.gcf()
	#plt.show()
	fig.savefig(filename)

wcloud(text, 'wc_all.png')

#now for pos, neu and neg respectively
pos_text = ' '.join([x for i,x in enumerate(tw_text) if tw_sent[i]=="positive"])
neu_text = ' '.join([x for i,x in enumerate(tw_text) if tw_sent[i]=="neutral"])
neg_text = ' '.join([x for i,x in enumerate(tw_text) if tw_sent[i]=="negative"])

wcloud(pos_text,'wc_pos.png')
wcloud(neu_text,'wc_neu.png')
wcloud(neg_text,'wc_neg.png')

#set up Stanford NLP: 
#from nltk.tag import StanfordNERTagger
#st = StanfordNERTagger('english.all.3class.distsim.crf.ser.gz') 
#print st.tag('Rami Eid is studying at Stony Brook University in NY'.split()) 
from nltk.tag import StanfordPOSTagger
st = StanfordPOSTagger('english-bidirectional-distsim.tagger') 
st.tag('What is the airspeed of an unladen swallow ?'.split())