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

documents=[]
allowed_word_types=["J","R"]
all_words=[]
for r in short_pos.split('\n'):
    documents.append((r,"pos"))
    words=word_tokenize(r)
    pos=nltk.pos_tag(words)
    for w in pos:
          if(w[1][0] in allowed_word_types):
                all_words.append(w[0].lower())
                
          

for r in short_neg.split('\n'):
    documents.append((r,"neg"))
    words=word_tokenize(r)
    pos=nltk.pos_tag(words)
    for w in pos:
          if(w[1][0] in allowed_word_types):
                all_words.append(w[0].lower())

save_documents=open("pickled_algos/documents.pickle","wb");
pickle.dump(documents,save_documents)
save_documents.close()


all_words=nltk.FreqDist(all_words)
words_features=list(all_words.keys())[:3000]


save_words_features=open("pickled_algos/words_features3k.pickle","wb");
pickle.dump(words_features,save_words_features)
save_words_features.close()


def find_features(document):    
    words=word_tokenize(document)
    features={}
    for w in words_features:
        features[w]=(w in words)

    return features

feature_sets=[(find_features(rev),category) for (rev,category) in documents ]

random.shuffle(feature_sets)

training_set=feature_sets[:3000]
test_set=feature_sets[3000:3200]

training_set=numpy.array(training_set)
test_set=numpy.array(test_set)

NaiveBayes_classifier=nltk.NaiveBayesClassifier.train(training_set)
print("Naive Bayes Classifier accuracy example",nltk.classify.accuracy(NaiveBayes_classifier,test_set))
NaiveBayes_classifier.show_most_informative_features(15)

save_NaiveBayes_classifier=open("pickled_algos/NaiveBayes_classifier.pickle","wb");
pickle.dump(NaiveBayes_classifier,save_NaiveBayes_classifier)
save_NaiveBayes_classifier.close()


MNB_classifier=SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MultinomialNB Classifier accuracy example",nltk.classify.accuracy(MNB_classifier,test_set))

save_MNB_classifier=open("pickled_algos/MNB_classifier.pickle","wb");
pickle.dump(MNB_classifier,save_MNB_classifier)
save_MNB_classifier.close()


Bernoulli_classifier=SklearnClassifier(BernoulliNB())
Bernoulli_classifier.train(training_set)
print("BernoulliNB Classifier accuracy example",nltk.classify.accuracy(Bernoulli_classifier,test_set))

save_Bernoulli_classifier=open("pickled_algos/Bernoulli_classifier.pickle","wb");
pickle.dump(Bernoulli_classifier,save_Bernoulli_classifier)
save_Bernoulli_classifier.close()


LinearSVC_Classifier=SklearnClassifier(LinearSVC())
LinearSVC_Classifier.train(training_set)
print("LinearSVC Classifier accuracy example",nltk.classify.accuracy(LinearSVC_Classifier,test_set))

save_LinearSVC_Classifier=open("pickled_algos/LinearSVC_Classifier.pickle","wb");
pickle.dump(LinearSVC_Classifier,save_LinearSVC_Classifier)
save_LinearSVC_Classifier.close()


LogisticRegression_classifier=SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression Classifier accuracy example",nltk.classify.accuracy(LogisticRegression_classifier,test_set))

save_LogisticRegression_classifier=open("pickled_algos/LogisticRegression_classifier.pickle","wb");
pickle.dump(LogisticRegression_classifier,save_LogisticRegression_classifier)
save_LogisticRegression_classifier.close()




SGDClassifier_classifier=SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier Classifier accuracy example",nltk.classify.accuracy(SGDClassifier_classifier,test_set))


save_SGDClassifier_classifier=open("pickled_algos/SGDClassifier_classifier.pickle","wb");
pickle.dump(SGDClassifier_classifier,save_SGDClassifier_classifier)
save_SGDClassifier_classifier.close()



NuSVC_classifier=SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC Classifier accuracy example",nltk.classify.accuracy(NuSVC_classifier,test_set))


save_NuSVC_classifier=open("pickled_algos/NuSVC_classifier.pickle","wb");
pickle.dump(NuSVC_classifier,save_NuSVC_classifier)
save_NuSVC_classifier.close()




voted_classifier=VoteClassifier(NaiveBayes_classifier,NuSVC_classifier,Bernoulli_classifier,LogisticRegression_classifier,MNB_classifier,LinearSVC_Classifier,SGDClassifier_classifier,)
print("Voted Classifier Accuracy",nltk.classify.accuracy(voted_classifier,test_set))




def sentiment(text):
    feats = find_features(text)
    return voted_classifier.classify(feats),voted_classifier.confidence(feats)
