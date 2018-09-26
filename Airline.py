# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
# Importing the dataset
dataset = pd.read_csv('Airline.csv')
y=dataset.iloc[:,5].values
y=np.array(y)
x1=dataset.iloc[:,7].values
x2=dataset.iloc[:,14].values
x1=np.asarray(x1,dtype='str')
x2=np.asarray(x2,dtype='str')
Y=np.zeros((len(y),3))
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
yt = labelencoder.fit_transform(y)
onehotencoder = OneHotEncoder(categorical_features ='all')
yt=yt.reshape(-1,1)
yt = onehotencoder.fit_transform(yt).toarray()

x=[]
for i in range(0,len(x1)):
    x.append(x1[i]+x2[i])
# Cleaning the texts

corpus = []
for i in range(0, len(x)):
    review = re.sub('[^a-zA-Z]', ' ', x[i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in (set(stopwords.words('english')),'nan')]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 2200)
X = cv.fit_transform(corpus).toarray()

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, yt, test_size = 0.20, random_state = 0)
ytt=np.zeros((len(y_test),1))
for i in range(0,len(y_test)):
    t=y_test[i][0]
    ind=0
    for j in range(0,3):
        if(y_test[i][j]>t):
            t=y_test[i][j]
            ind=j
    ytt[i]=ind
import keras
from keras.models import Sequential
from keras.layers import Dense
classifier = Sequential()
classifier.add(Dense(output_dim = 2200, init = 'uniform', activation = 'relu', input_dim = 2200))
classifier.add(Dense(output_dim = 2200, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 1100, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 1100, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 3, init = 'uniform', activation = 'softmax'))
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train, batch_size = 20, nb_epoch = 50)
y_pred = classifier.predict(X_test)
ypt=np.zeros((len(y_pred),1))
for i in range(0,len(y_pred)):
    t=y_pred[i][0]
    ind=0
    for j in range(0,3):
        if(y_pred[i][j]>t):
            t=y_pred[i][j]
            ind=j
    ypt[i]=ind
        
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ytt, ypt)
