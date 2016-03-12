#!/usr/bin/python2.7
from sklearn.cross_validation import train_test_split
from gensim.models.word2vec import Word2Vec
from sklearn.preprocessing import scale, MinMaxScaler
from sklearn.linear_model import SGDClassifier, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.lda import LDA
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import os 
import re
from string import punctuation
# import tsne
import seaborn as sns

def cleanText(corpus, stopWords):
	#corpus = [re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',z) for z in corpus]
	#corpus = [re.sub('@[^\s]+','AT_USER',z) for z in corpus]
	#corpus = [re.sub('[\s]+', ' ', z) for z in corpus]
	#corpus = [re.sub(r'#([^\s]+)', r'\1', z) for z in corpus]

	# trim " 
	corpus = [z.strip('\'"') for z in corpus] 
	corpus = [z.lower().replace('\n','') for z in corpus]

	corpus = [[y.strip(punctuation) for y in z.split()] for z in corpus]

	# for i,z in enumerate(corpus):
	# 	corpus[i] = [w for w in z if w not in stopWords ]

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

def plotlen(alltw, filltw):
	n = len(alltw)
	c = sns.color_palette()

	ind = np.arange(n)
	width = 0.35
	fig, ax = plt.subplots()
	rects1 = ax.bar(ind, alltw, width, color=c[1])
	rects2 = ax.bar(ind+width, filltw, width, color=c[2])

	ax.set_ylabel('count')
	ax.set_title('count of sentiments and confidence')
	ax.set_xticks(ind + width)
	ax.set_xticklabels(('negative','positive', 'neutral'))

	ax.legend((rects1[0], rects2[0]), ('all tweets', 'filtered tweets'))

	plt.show()

def param_optimization(clf, X_train):
	# optimization with grid search
	plt.xlabel('alpha')
	plt.ylabel('accuracy')
	plt.title('MultinomialNB')
	plt_x = []
	plt_y = []
	plt2_x = []
	plt2_y = []
	print('\n-------------- Optimization --------------')
	parameters = {'fit_prior':(True, False), 'alpha':[0, .2, .4, .6, .8, 1, 1.2, 1.4, 1.6, 1.8, 2.0]}
	clf = GridSearchCV(clf, parameters, cv=5)
	# clf.fit([extract_features(w) for w in X_train], Y_train)
	clf.fit(X_train, Y_train)
	print "Best accuracy score for MultinomialNB:"
	print clf.best_score_, 'with', clf.best_params_
	print "\nGrid scores on development set:"
	for params, mean_score, scores in clf.grid_scores_:
	    print("%0.3f (+/-%0.03f) for %r"
	          % (mean_score, scores.std() * 2, params))
	    if params['fit_prior'] == True:
	    	plt_x.append(params['alpha'])
	    	plt_y.append(mean_score)
	    else:
	   		plt2_x.append(params['alpha'])
	   		plt2_y.append(mean_score)
	print '\nInitializing Multinomial Naive Bayes with fit_prior={} and alpha={}'.format(clf.best_params_.values()[1], clf.best_params_.values()[0])
	opt_mnb = MultinomialNB(fit_prior=clf.best_params_.values()[1], alpha=clf.best_params_.values()[0])
	print '-------------- End Optimization --------------\n'
	plt.plot(plt_x, plt_y, label='fit prior')
	plt.plot(plt2_x, plt2_y, 'r', label='uniform prior')
	plt.legend(loc=4)
	plt.show()
	return opt_mnb


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


	pos_text = tweets.text[tw_sent == "positive"]#+tw_air[tweets.airline_sentiment == "positive"]
	neu_text = tweets.text[tw_sent == "neutral" ]#+tw_air[tweets.airline_sentiment == "neutral"] 
	neg_text = tweets.text[tw_sent == "negative"]#+tw_air[tweets.airline_sentiment == "negative"]

	alltw = [len(neg_text),len(pos_text),len(neu_text)]
	#print "pos:",len(pos_text), "neu:", len(neu_text), "neg:" , len(neg_text)

	pos_text = [t for i,t in enumerate(tweets.text) if tw_sent[i] == "positive" and tw_sent_conf[i] == 1.0]
	neu_text = [t for i,t in enumerate(tweets.text) if tw_sent[i] == "neutral"  and tw_sent_conf[i] == 1.0]
	neg_text = [t for i,t in enumerate(tweets.text) if tw_sent[i] == "negative" and tw_sent_conf[i] == 1.0]

	filtw = [len(neg_text),len(pos_text),len(neu_text)]
	#print "pos:", len(pos_text), "neu:", len(neu_text), "neg:" ,len(neg_text)

	# plotlen(alltw, filtw)
	

	#use 1 for positive sentiment, -1 for negative, 0 for neutral
	y = np.concatenate((np.ones(len(pos_text)), np.zeros(len(neu_text)), np.full(len(neg_text),-1, dtype=np.int)))
	#y = np.concatenate((np.ones(len(pos_text)), np.zeros(len(neu_text))))
	#y = np.concatenate((np.ones(len(pos_text)), np.zeros(len(neg_text))))
	#y = np.concatenate((np.zeros(len(neg_text)), np.ones(len(neu_text))))

	x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos_text, neu_text, neg_text)), y, test_size=0.4, random_state=0)
	#x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos_text, neu_text)), y, test_size=0.2, random_state=0)
	#x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos_text, neg_text)), y, test_size=0.2, random_state=0)
	#x_train, x_test, y_train, y_test = train_test_split(np.concatenate((neu_text, neg_text)), y, test_size=0.2, random_state=0)

	stopWords = getStopWordList('stopwords.txt')

	x_train = cleanText(x_train, stopWords)
	x_test = cleanText(x_test, stopWords)

	#print x_train

	n_dim = 400
	#Initialize model and build vocab
	#min_count = ignore all words with total frequency lower than this.
	tw_w2v = Word2Vec(size=n_dim, min_count=50, negative=0, window=5) #size = size of the NN layer
	tw_w2v.build_vocab(x_train)

	#Train the model 
	tw_w2v.train(x_train)

	print tw_w2v.most_similar('good')

	train_vecs = np.concatenate([buildWordVector(z, n_dim) for z in x_train])
	train_vecs = scale(train_vecs)

	#y = tsne.tsne(train_vecs, 2)


	#Train word2vec on test tweets
	#tw_w2v.train(x_test)

	#Build test tweet vectors then scale
	test_vecs = np.concatenate([buildWordVector(z, n_dim) for z in x_test])
	test_vecs = scale(test_vecs)

	

	# lr = SGDClassifier(loss='log', penalty='l1')
	# parameters = {'loss':('log', 'hinge', 'squared_hinge', 'modified_huber'),
	# 			 'alpha':[.3,.6,1,1.3,1.6,2],
	# 			 'penalty':['none', 'l2', 'l1', 'elasticnet']}
	# clf = GridSearchCV(lr, parameters, cv=5)
	# # clf.fit([extract_features(w) for w in X_train], Y_train)
	# clf.fit(train_vecs, y_train)
	# print "Best accuracy score:"
	# print clf.best_score_, 'with', clf.best_params_
	# print "\nGrid scores on development set:"
	# for params, mean_score, scores in clf.grid_scores_:
	#     print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params))
	# lr = SGDClassifier(loss=clf.best_params_['loss'], alpha=clf.best_params_['alpha'], penalty=clf.best_params_['penalty'])	

	lr = RidgeClassifier() # 85
	parameters = {'normalize':(True, False),
				 'alpha':[.3,.6,1,1.3,1.6,2],
				 'solver':['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg']}

	clf = GridSearchCV(lr, parameters, cv=5)
	clf.fit(train_vecs, y_train)
	print "Best accuracy score:"
	print clf.best_score_, 'with', clf.best_params_
	print "\nGrid scores on development set:"
	for params, mean_score, scores in clf.grid_scores_:
	    print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params))
	lr = RidgeClassifier(normalize=clf.best_params_['normalize'], alpha=clf.best_params_['alpha'], solver=clf.best_params_['solver'])
	

	# lr = LDA() # 85
	# parameters = {
	# 			 'tol':[1e-06, 1e-05, 1e-04, 1e-03, 1e-02, 1e-01],
	# 			 'solver':['svd', 'lsqr']}

	# clf = GridSearchCV(lr, parameters, cv=5)
	# clf.fit(train_vecs, y_train)
	# print "Best accuracy score:"
	# print clf.best_score_, 'with', clf.best_params_
	# print "\nGrid scores on development set:"
	# for params, mean_score, scores in clf.grid_scores_:
	#     print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params))
	# lr = LDA(tol=clf.best_params_['tol'], solver=clf.best_params_['solver'])

	# lr = MultinomialNB()
	# mms= MinMaxScaler(feature_range=(0,1))
	# train_vecs = mms.fit_transform(train_vecs)
	# parameters = {'fit_prior':(True, False), 'alpha':[0, .2, .4, .6, .8, 1, 1.2, 1.4, 1.6, 1.8, 2.0]}
	# clf = GridSearchCV(lr, parameters, cv=5)
	# # clf.fit([extract_features(w) for w in X_train], Y_train)
	# clf.fit(train_vecs, y_train)
	# print "Best accuracy score for MultinomialNB:"
	# print clf.best_score_, 'with', clf.best_params_
	# print "\nGrid scores on development set:"
	# for params, mean_score, scores in clf.grid_scores_:
	#     print("%0.3f (+/-%0.03f) for %r"
	#           % (mean_score, scores.std() * 2, params))
	# lr = MultinomialNB(fit_prior=clf.best_params_.values()[1], alpha=clf.best_params_.values()[0])


	lr.fit(train_vecs, y_train)



	# print 'Test Accuracy: %.2f'%lr.score(mms.transform(test_vecs), y_test)
	print 'Test Accuracy: %.2f'%lr.score(test_vecs, y_test)

	# for i in xrange(20):
	# 	print lr.predict(test_vecs[i]), x_test[i]
	# # print "\n",tw_w2v.most_similar('wonderful')
	# # print "\n",tw_w2v.most_similar('bad')



	#############################plot ROC #########################################

	# n_classes = 3
	# pred_probas = lr.predict_proba(test_vecs)[:,1]
	# print pred_probas.shape
	# print y_test.shape

	# plt.figure()

	# # fpr = dict()
	# # tpr = dict()
	# # roc_auc = dict()

	# # for i in range(n_classes):
	# # 	fpr[i], tpr[i], _ = roc_curve(y_test, y_score[:, i])
	# # 	roc_auc[i] = auc(fpr[i], tpr[i])

	# # for i in range(n_classes):
	# # 	plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
	# # 								   ''.format(i, roc_auc[i]))

	# fpr,tpr,_ = roc_curve(y_test, pred_probas)
	# roc_auc = auc(fpr,tpr)
	# plt.plot(fpr,tpr,label='area = %.2f' %roc_auc)

		
	# plt.plot([0, 1], [0, 1], 'k--')
	# plt.xlim([0.0, 1.0])
	# plt.ylim([0.0, 1.05])
	# plt.xlabel('False Positive Rate')
	# plt.ylabel('True Positive Rate')
	# plt.legend(loc='lower right')
		
	# plt.show()

	#########################################

