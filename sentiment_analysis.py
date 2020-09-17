#importing the libraries
import numpy as np
import pandas as pd

#importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv' , delimiter = '\t' , quoting = 3)

#cleaning the text
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range (0,1000):
    #replacing characters except a-z by space
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    #turing all characters to lowercase
    review = review.lower()
    #spliting the sentences into words
    review = review.split()
    #stemming (loved = love)
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)

#creating the bag of word model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
x = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,-1].values

#spliting into test and train set
from sklearn.model_selection import train_test_split
x_train , x_test, y_train , y_test = train_test_split(x,y,test_size = 0.20,random_state =0)

#training the naive bayes model
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train,y_train)

#predicting the results
y_pred = classifier.predict(x_test)

#making confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)
