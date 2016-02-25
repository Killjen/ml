#!/usr/bin/python2.7
from sklearn.cross_validation import train_test_split
from gensim.models.word2vec import Word2Vec
from sklearn.preprocessing import scale
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import os 
import re


def cleanText(corpus, stopWords):
	corpus = [re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',z) for z in corpus]
	corpus = [re.sub('@[^\s]+','AT_USER',z) for z in corpus]
	corpus = [re.sub('[\s]+', ' ', z) for z in corpus]
	corpus = [re.sub(r'#([^\s]+)', r'\1', z) for z in corpus]
	corpus = [z.strip('\'"') for z in corpus]
	corpus = [z.lower().replace('\n','') for z in corpus]

	corpus = [z.split() for z in corpus]

	for i,z in enumerate(corpus):
		corpus[i] = [w for w in z if w not in stopWords ]

	return corpus

def getStopWordList(stopWordListFileName):
	#read the stopwords file and build a list
	stopWords = []
	stopWords.append('at_user')
	stopWords.append('url')

	fp = open(stopWordListFileName, 'r')
	line = fp.readline()
	while line:
		word = line.strip()
		stopWords.append(word)
		line = fp.readline()
	fp.close()
	return stopWords


#Build word vector for training set by using the average value of all word vectors in the tweet, then scale
def buildWordVector(text, size):
	vec = np.zeros(size).reshape((1, size))
	count = 0.
	for word in text:
		try:
			vec += tw_w2v[word].reshape((1, size))
			count += 1.
		except KeyError:
			continue
	if count != 0:
		vec /= count

	return vec


if __name__ == '__main__':

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

	pos_text = tweets.text[tweets.airline_sentiment == "positive"]#+tw_air[tweets.airline_sentiment == "positive"]
	neu_text = tweets.text[tweets.airline_sentiment == "neutral"]#+tw_air[tweets.airline_sentiment == "neutral"] 
	neg_text = tweets.text[tweets.airline_sentiment == "negative"]#+tw_air[tweets.airline_sentiment == "negative"]


	#use 1 for positive sentiment, -1 for negative, 0 for neutral
	#y = np.concatenate((np.ones(len(pos_text)), np.zeros(len(neu_text)), np.full(len(neg_text),-1, dtype=np.int)))
	#y = np.concatenate((np.ones(len(pos_text)), np.zeros(len(neu_text))))
	y = np.concatenate((np.ones(len(pos_text)), np.zeros(len(neg_text))))
	#y = np.concatenate((np.zeros(len(neg_text)), np.ones(len(neu_text))))

	#x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos_text, neu_text, neg_text)), y, test_size=0.2, random_state=0)
	#x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos_text, neu_text)), y, test_size=0.2, random_state=0)
	x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos_text, neg_text)), y, test_size=0.2, random_state=0)
	#x_train, x_test, y_train, y_test = train_test_split(np.concatenate((neu_text, neg_text)), y, test_size=0.2, random_state=0)

	stopWords = getStopWordList('stopwords.txt')

	x_train = cleanText(x_train, stopWords)
	x_test = cleanText(x_test, stopWords)

	#print x_train

	n_dim = 300
	#Initialize model and build vocab
	tw_w2v = Word2Vec(size=n_dim, min_count=10) #size = size of the NN layer
	tw_w2v.build_vocab(x_train)

	#Train the model 
	tw_w2v.train(x_train)


	train_vecs = np.concatenate([buildWordVector(z, n_dim) for z in x_train])
	train_vecs = scale(train_vecs)


	#Train word2vec on test tweets
	#tw_w2v.train(x_test)

	#Build test tweet vectors then scale
	test_vecs = np.concatenate([buildWordVector(z, n_dim) for z in x_test])
	test_vecs = scale(test_vecs)
	

	lr = SGDClassifier(loss='log', penalty='l1', shuffle=False)
	lr.fit(train_vecs, y_train)

	print 'Test Accuracy: %.2f'%lr.score(test_vecs, y_test)

	# for i in xrange(20):
	# 	print lr.predict(test_vecs[i]), x_test[i]
	print "\n",tw_w2v.most_similar('wonderful')


	#ROC curve geht nur bei positiv negative (ohne neutral)
	pred_probas = lr.predict_proba(test_vecs)[:,1]

	fpr,tpr,_ = roc_curve(y_test, pred_probas)
	roc_auc = auc(fpr,tpr)
	plt.plot(fpr,tpr,label='area = %.2f' %roc_auc)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.legend(loc='lower right')
		
	plt.show()
