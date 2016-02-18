from sklearn.cross_validation import train_test_split
from gensim.models.word2vec import Word2Vec
from sklearn.preprocessing import scale
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import os 

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

pos_text = tweets.text[tweets.airline_sentiment == "positive"]
neu_text = tweets.text[tweets.airline_sentiment == "neutral"]
neg_text = tweets.text[tweets.airline_sentiment == "negative"]

#use 1 for positive sentiment, -1 for negative, 0 for neutral
y = np.concatenate((np.ones(len(pos_text)), np.zeros(len(neu_text)), np.full(len(neg_text),-1,dtype=np.int)))

x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos_text, neu_text, neg_text)), y, test_size=0.2)

def cleanText(corpus):
    corpus = [z.lower().replace('\n','').split() for z in corpus]
    return corpus

x_train = cleanText(x_train)
x_test = cleanText(x_test)

n_dim = 300
#Initialize model and build vocab
tw_w2v = Word2Vec(size=n_dim, min_count=10) #size = size of the NN layer
tw_w2v.build_vocab(x_train)

#Train the model 
tw_w2v.train(x_train)

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

train_vecs = np.concatenate([buildWordVector(z, n_dim) for z in x_train])
train_vecs = scale(train_vecs)

#Train word2vec on test tweets
#tw_w2v.train(x_test)

#Build test tweet vectors then scale
test_vecs = np.concatenate([buildWordVector(z, n_dim) for z in x_test])
test_vecs = scale(test_vecs)

lr = SGDClassifier(loss='log', penalty='l1')
lr.fit(train_vecs, y_train)

print 'Test Accuracy: %.2f'%lr.score(test_vecs, y_test)


#ROC curve geht nur bei positiv negative (ohne neutral)
# pred_probas = lr.predict_proba(test_vecs)[:,1]

# fpr,tpr,_ = roc_curve(y_test, pred_probas)
# roc_auc = auc(fpr,tpr)
# plt.plot(fpr,tpr,label='area = %.2f' %roc_auc)
# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.legend(loc='lower right')
	
# plt.show()
