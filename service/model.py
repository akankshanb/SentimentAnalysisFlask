import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import re 
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pickle

def train():
    dataset = pd.read_csv('data/Restaurant_Reviews.csv', delimiter = '\t' , quoting = 3)
    y = dataset.iloc[:,1].values
    labelencoder_y = LabelEncoder()
    y = labelencoder_y.fit_transform(y) 
    x,cv = bow(dataset)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)
    classifier = BernoulliNB()
    classifier.fit(X_train, y_train)
    
    filename = 'NB_model.pkl'
    pickle.dump(classifier, open(filename, 'wb'))
    NB_model = pickle.load(open(filename, 'rb'))

    y_pred = NB_model.predict(X_test)
    y_prob = NB_model.predict_proba(X_test)

    cm = confusion_matrix(y_test, y_pred)
    return cv, NB_model

def bow(dataset):
    corpus = []
    for i in range(len(dataset['Review'])):
        review = re.sub('[^a-zA-Z]', ' ' ,  dataset['Review'][i]).lower()
        review = review.split()
        ps = PorterStemmer()
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review = ' '.join(review)
        corpus.append(review)

    cv = CountVectorizer(max_features = 1500)
    X = cv.fit_transform(corpus).toarray() #sparse matrix
    return X, cv

    
