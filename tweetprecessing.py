import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re#regular expression
import nltk
from nltk.corpus import stopwords
#nltk.download('stopwords')
#stopwords=stopwords.words('english')
from nltk.stem.porter import PorterStemmer#stemming is used to shorten up the lookup and normalize sentences
ps=PorterStemmer()

dataset=pd.read_csv("F:/dataset/train.csv")
#dataset['tweet'][0]

processed_tweet=[]

for i in range(31962):
    
    tweet =re.sub('@[\w]*',' ',dataset['tweet'][i])#here all the @user will be removed
    tweet =re.sub('[^a-zA-Z#]', ' ',dataset['tweet'][i]) #emojies
    tweet=tweet.lower()#this will lower the text of each tweeet
    tweet=tweet.split()#changing tweets in list
    tweet=[ps.stem(token) for token in tweet if not token in stopwords.words('english')]#token is the smallest no.so wo saare words jo stopwords me nhi hai
    tweet=' '.join(tweet)
    processed_tweet.append(tweet)  #here all the tweets that will be filtered will enter here          

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 3000)
X = cv.fit_transform(processed_tweet)
X =  X.toarray()
y = dataset.iloc[:, 1].values
print(cv.get_feature_names())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

log_reg.score(X_train, y_train)
log_reg.score(X_test, y_test)
log_reg.score(X, y)

 