#!/usr/bin/python2.7
import numpy as np
import matplotlib as mplt
import matplotlib.pyplot as plt
import os
import pandas as pd
import math
from wordcloud import WordCloud
import re
import nltk 
from random import shuffle

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

def getStats():
	'''
	Stats 
	'''
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

#wcloud(text, 'wc_all.png')

#now for pos, neu and neg respectively
pos_text = ' '.join([x for i,x in enumerate(tw_text) if tw_sent[i]=="positive"])
neu_text = ' '.join([x for i,x in enumerate(tw_text) if tw_sent[i]=="neutral"])
neg_text = ' '.join([x for i,x in enumerate(tw_text) if tw_sent[i]=="negative"])

# wcloud(pos_text,'wc_pos.png')
# wcloud(neu_text,'wc_neu.png')
# wcloud(neg_text,'wc_neg.png')

#set up Stanford NLP: 
#from nltk.tag import StanfordNERTagger
#st = StanfordNERTagger('english.all.3class.distsim.crf.ser.gz') 
#print st.tag('Rami Eid is studying at Stony Brook University in NY'.split()) 
# from nltk.tag import StanfordPOSTagger
# st = StanfordPOSTagger('english-bidirectional-distsim.tagger') 
# st.tag('What is the airspeed of an unladen swallow ?'.split())

#start process_tweet
def processTweet(tweet):
	# process the tweets

	#Convert to lower case
	tweet = tweet.lower()
	#Convert www.* or https?://* to URL
	tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
	#Convert @username to AT_USER
	tweet = re.sub('@[^\s]+','AT_USER',tweet)
	#Remove additional white spaces
	tweet = re.sub('[\s]+', ' ', tweet)
	#Replace #word with word
	tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
	#trim
	tweet = tweet.strip('\'"')
	return tweet
#end


#initialize stopWords
stopWords = []


def replaceTwoOrMore(s):
	#look for 2 or more repetitions of character and replace with the character itself
	pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
	return pattern.sub(r"\1\1", s)


def getStopWordList(stopWordListFileName):
	#read the stopwords file and build a list
	stopWords = []
	stopWords.append('AT_USER')
	stopWords.append('URL')

	fp = open(stopWordListFileName, 'r')
	line = fp.readline()
	while line:
		word = line.strip()
		stopWords.append(word)
		line = fp.readline()
	fp.close()
	return stopWords


def getFeatureVector(tweet, stopWords):
	featureVector = []
	#split tweet into words
	words = tweet.split()
	for w in words:
		#replace two or more with two occurrences
		w = replaceTwoOrMore(w)
		#strip punctuation
		w = w.strip('\'"?,.')
		#check if the word stats with an alphabet
		val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
		#ignore if it is a stop word
		if(w in stopWords or val is None):
			continue
		else:
			featureVector.append(w.lower())
	return featureVector


st = open('stopwords.txt', 'r')
stopWords = getStopWordList('stopwords.txt')

# processed_Tweet = tw_text.copy()
# for x in xrange(len(tw_text)):
# 	processed_Tweet[x] = processTweet(tw_text[x])
# 	featureVector = getFeatureVector(processed_Tweet[x])
# 	print featureVector

#get training and test set
#first shuffle tweets
rp = np.random.permutation(n)

#training data: first 2/3 of shuffled tweets
train_set = [(tw_text[i],tw_sent[i]) for j,i in enumerate(rp) if j<(2.0/3*n) and tw_sent_conf[i] == 1 and tw_sent[i] != "neutral"] #for now: only allow conf. of 1 and no neutral
test_set = [(tw_text[i],tw_sent[i]) for j,i in enumerate(rp) if j>=(2.0/3*n) and tw_sent_conf[i] == 1 and tw_sent[i] != "neutral"]

train_set = map(lambda x : (processTweet(x[0]),x[1]), train_set)
test_set = map(lambda x : (processTweet(x[0]),x[1]), test_set) #HASKELL!

#get the featurevector
train_set = map(lambda x : (getFeatureVector(x[0], stopWords),x[1]), train_set)
test_set = map(lambda x : (getFeatureVector(x[0], stopWords),x[1]), test_set)

featureList = []   #List of Words, if a tweet contains word at index m the mth entry of the featurevector will be the number of occurences of that word
for (words,sent) in train_set:
	featureList.extend(words)
print len(featureList)

# Remove featureList duplicates
featureList = list(set(featureList))
print len(featureList)


def extract_features(tweet_words):
	featureVec = np.zeros(len(featureList))
	for word in tweet_words:
		for i in xrange(len(featureList)):
			if word == featureList[i]: #Occurences
				featureVec[i]+=1
	#for (i,word) in enumerate(featureList):
	#	featureVec[i] = (word in tweet_words) 
	return featureVec
#

def test_class(nb):
	# Test the classifier
	y_real = [s for (w,s) in test_set]
	y_pred = gnb.fit([extract_features(w) for (w,s) in train_set], [s for (w,s) in train_set]).predict([extract_features(w) for (w,s) in test_set])

	def eval(real,pred,sent="none"):
		correct = 0
		if sent == "none":
			total = len(y_real)
			for i in xrange(len(y_real)):
				if  y_pred[i] == y_real[i]:
					correct += 1
		else:
			total = 0
			for i in xrange(len(y_real)):
				if y_real[i] == sent:
					total += 1
					if  y_pred[i] == sent:
						correct += 1

		print "GNBayes: Correctly predicted " + str(correct/(1.0*total)) + " for sent " + sent

	eval(y_real,y_pred)
	eval(y_real,y_pred,"positive")
	#eval(y_real,y_pred,"neutral")
	eval(y_real,y_pred,"negative")
	# print informative features about the classifier

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

print " "
print "Gaussian NB: "
test_class(gnb)