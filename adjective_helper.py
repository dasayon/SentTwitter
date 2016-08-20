import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.svm import LinearSVC,SVC,NuSVC
from sklearn.linear_model import LogisticRegression,SGDClassifier
import pickle
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize
import numpy

class VoteClassifier(ClassifierI):
      def __init__(self,*classifiers):
        self.classifiers=classifiers

      def classify(self, features):
          votes=[]
          for c in self.classifiers:
            v=c.classify(features)
            votes.append(v)
          return mode(votes)


      def confidence(self,features):
        votes=[]
        for c in self.classifiers:
            v=c.classify(features)
            votes.append(v)
        choice_votes=votes.count(mode(votes))
        conf=choice_votes/len(votes)
        return conf

short_pos=open("short_reviews/positive.txt","r").read()
short_neg=open("short_reviews/negative.txt","r").read()


allowed_word_types=["J"]
all_words=[]
for r in short_pos.split('\n'):
    words=word_tokenize(r)
    pos=nltk.pos_tag(words)
    for w in pos:
          if(w[1][0] in allowed_word_types):
                all_words.append(w[0].lower())
                
          

for r in short_neg.split('\n'):
    words=word_tokenize(r)
    pos=nltk.pos_tag(words)
    for w in pos:
          if(w[1][0] in allowed_word_types):
                all_words.append(w[0].lower())


documents_f = open("pickled_algos/documents.pickle", "rb")
documents = pickle.load(documents_f)
documents_f.close()


all_words=nltk.FreqDist(all_words)
words_features=list(all_words.keys())[:3000]


save_words_features=open("pickled_algos/words_features_adjective.pickle","wb");
pickle.dump(words_features,save_words_features)
save_words_features.close()


