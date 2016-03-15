#!/usr/bin/python2.7
import numpy as np
import matplotlib as mplt
import matplotlib.pyplot as plt
import os
import pandas as pd
import math
from wordcloud import WordCloud
import re
import emoji #https://pypi.python.org/pypi/emoji/
import time

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

ex = 10246 #example tweet to show preprocessing

print tw_text[ex]
print tw_air[ex]
print tw_loc[ex]
print tw_coord[ex]

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
	smileys = 0
	hashtags = 0
	for i in xrange(n):
		if type(tw_coord[i]) is str:
			coord += 1
		if type(tw_loc[i]) is str:
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
		utwt = unicode(tw_text[i], "utf-8")
		if ":)" in tw_text[i] or ":-)" in tw_text[i] or ":D" in tw_text[i] or ":-D" in tw_text[i] or "=)" in tw_text[i] or "=D" in tw_text[i] or ";)" in tw_text[i] or ";-)" in tw_text[i] or ":'(" in tw_text[i] or ":-(" in tw_text[i] or ":(" in tw_text[i] or "D:" in tw_text[i] or "xD" in tw_text[i] or emoji.emojize(':thumps_up:', use_aliases=True) in utwt or emoji.emojize(':smile:', use_aliases=True) in utwt or emoji.emojize(':laughing:', use_aliases=True) in utwt or emoji.emojize(':smiley', use_aliases=True) in utwt or emoji.emojize(':smirk:', use_aliases=True) in utwt or emoji.emojize(':wink:', use_aliases=True) in utwt or emoji.emojize(':satisfied:', use_aliases=True) in utwt:
			smileys+=1  
		if len(re.findall("( #([^\s]+))", tw_text[i])) > 0:
			hashtags+=1

	print "available coords: " + str(coord) + " out of " + str(n) + " (" + str(100*coord/(1.0*n)) + "%)"
	print "available locs: " + str(loc) + " out of " + str(n) + " (" + str(100*loc/(1.0*n)) + "%)"
	print "#RT > 0: " + str(rt) + " out of " + str(n) + " (" + str(100*rt/(1.0*n)) + "%)"
	print "conf < 1: " + str(conf) + " out of " + str(n) + " (" + str(100*conf/(1.0*n)) + "%)"
	print "sentiment positive: " + str(pos) + " out of " + str(n) + " (" + str(100*pos/(1.0*n)) + "%)"
	print "sentiment neutral: " + str(neu) + " out of " + str(n) + " (" + str(100*neu/(1.0*n)) + "%)"
	print "sentiment negative: " + str(neg) + " out of " + str(n) + " (" + str(100*neg/(1.0*n)) + "%)"
	print "smileys: " + str(smileys) + " out of " + str(n) + " (" + str(100*smileys/(1.0*n)) + "%)"
	print "hashtags: " + str(hashtags) + " out of " + str(n) + " (" + str(100*hashtags/(1.0*n)) + "%)"

getStats()

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
#pos_text = ' '.join([x for i,x in enumerate(tw_text) if tw_sent[i]=="positive"])
#neu_text = ' '.join([x for i,x in enumerate(tw_text) if tw_sent[i]=="neutral"])
#neg_text = ' '.join([x for i,x in enumerate(tw_text) if tw_sent[i]=="negative"])

# wcloud(pos_text,'wc_pos.png')
# wcloud(neu_text,'wc_neu.png')
# wcloud(neg_text,'wc_neg.png')

############### Begin with NB Classification #######################################

#read stop words from the file into list
stop_words = []
with open('stopwords.txt','r') as f:
    for line in f:
        for word in line.split():
           stop_words.append(word) 

def getWords(tweet, stop_words): #preprocessing
	tweet = tweet.lower()
	words = []
	#split tweet into words
	tw_words = tweet.split()
	for w in tw_words:
		#ignore @username mentions:
		if(w[0] == '@'):
			continue
		#turn an URL into 'url', regex from http://stackoverflow.com/questions/6038061/regular-expression-to-find-urls-within-a-string
		w = re.sub('(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?','url',w)
		#remove special characters
		w = ''.join(c for c in w if c.isalnum())
		#ignore  stop words and numbers
		if(w.lower() in stop_words or w.isdigit()):
			continue
		else:
			words.append(w.lower())
		#words.append(w.lower())
	return words


#show preprocessing on the example
example = tw_text[ex] + " " + tw_air[ex] + " " + tw_loc[ex]
print getWords(example, stop_words)

## preprocess the data
#treat airline as a word
prep_set = [(tw_text[i] + " " + tw_air[i],tw_loc[i],tw_sent[i]) for i in xrange(n)] # if tw_sent[i] != "neutral" and tw_sent_conf[i] == 1] #for now: only allow conf. of 1 and no neutral

#treat location as a word if available:
prep_set = map(lambda x : (x[0] + " " + x[1],x[2]) if type(x[1]) is str else (x[0],x[2]) , prep_set)

#get the featurevector as vector of words
prep_set = map(lambda x : (getWords(x[0], stop_words),x[1]), prep_set)

#split into training and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split([x for (x,s) in prep_set], [s for (x,s) in prep_set], test_size=0.4, random_state=0)


featureList = []   #List of Words, if a tweet contains the word featureList[m] the mth entry of the featurevector will be the number of occurences of that word
for words in X_train:
	featureList.extend(words)
#len(featureList)

# Remove featureList duplicates
featureList = list(set(featureList))
print "#features: ", len(featureList)

def getFeatures(tweet_words):
	featureVec = np.zeros(len(featureList))
	for word in tweet_words:
		for i in xrange(len(featureList)):
			if word == featureList[i]: #Occurences
				featureVec[i]+=1
	return featureVec

#
def eval(real,pred,sent="none"):
	correct = 0
	if sent == "none":
		total = len(real)
		for i in xrange(len(real)):
			if  pred[i] == real[i]:
				correct += 1
	else:
		total = 0
		for i in xrange(len(real)):
			if real[i] == sent:
				total += 1
				if  pred[i] == sent:
					correct += 1

	print "MNBayes: Correctly predicted " + str(correct/(1.0*total)) + " for sent " + sent

def test_class(nb):
	start = time.time()
	# Test the classifier
	Y_pred = nb.fit([getFeatures(w) for w in X_train], Y_train).predict([getFeatures(w) for w in X_test])
	end = time.time()
	diff = end - start
	print "took " + str(int(diff / 60)) + ":" + str(int(diff % 60)) + " minutes"

	eval(Y_test,Y_pred)
	eval(Y_test,Y_pred,"positive")
	eval(Y_test,Y_pred,"neutral")
	eval(Y_test,Y_pred,"negative")
	# print informative features about the classifier

from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()

print " "
print "Multinomial NB: "
test_class(mnb)

'''#show with an example how mnb handles irony
ironic_tweet = "Great customer support I've only been waiting for 2 hours, thanks love it!"
normal_tweet = "Bad customer support I've been waiting for 2 hours, hate it!"
ironic_tweet2 = "Bad customer support I've been waiting for 1 second, totally hate it ;)"
normal_tweet2 = "Great customer support I've only been waiting for 1 second, thanks love it!"
normal_tweet3 = "Great customer support, thanks love it!"
all_tweets = [ironic_tweet, normal_tweet, ironic_tweet2, normal_tweet2, normal_tweet3]
print "ironic_tweet prediction: ", mnb.fit([getFeatures(w) for w in X_train], Y_train).predict([getFeatures(getWords(tweet, stop_words)) for tweet in all_tweets])
'''
def print_top15(feature_names,clf):
    """Prints features with the highest coefficient values, per class"""
    class_labels=clf.classes_
    for i, class_label in enumerate(class_labels):
        top10 = np.argsort(clf.coef_[i])[-15:]
        print("%s: %s" % (class_label,
              " ".join(feature_names[j] for j in top10)))

print_top15(featureList,mnb)