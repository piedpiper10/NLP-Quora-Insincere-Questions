from sklearn.metrics import f1_score
from sklearn import naive_bayes
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as numpy
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
train = pd.read_csv('train_75.csv')
test = pd.read_csv('valid_25.csv')

sentences1 = numpy.array(train['question_text'])
sentences2 = numpy.array(test['question_text'])
sentences = numpy.concatenate((sentences1, sentences2), axis=0)
vectorizer = CountVectorizer(stop_words='english')
X_train= vectorizer.fit_transform(train['question_text'])
X_test= vectorizer.transform(test['question_text'])

classifier = naive_bayes.MultinomialNB()
classifier=LogisticRegression()
classifier.fit(X_train, train['target'])

predictions = classifier.predict(X_test)
print f1_score(test['target'],predictions)
'''
USING THE TFIDFVECTOR
vectorizer =TfidfVectorizer()
vectorizer.fit(sentences)
X_train = vectorizer.transform(train['question_text'])
X_test = vectorizer.transform(test['question_text'])
print X_train, X_test
classifier = naive_bayes.MultinomialNB()
classifier.fit(X_train, train['target'])

predictions = classifier.predict(X_test)
print f1_score(test['target'],predictions)
'''

