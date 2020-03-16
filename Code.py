#*********************************

########import the necessary libraries

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Dropout, Bidirectional
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
import re


########load and clean the training data

def clean_model_data(path):
    #read data
    data = pd.read_csv(path, encoding= 'iso-8859-1')
    data['Phrase'] = data['Phrase'].str.encode('utf8').str.decode('ascii', 'ignore')
    data['Phrase'] = data['Phrase'].str.replace('[^\w\s]','')
    
    #retrieve column of positive or negative sentiments
    data['sentiment'] = np.where(data['sentiment_values'] >= 0.5, 1, 0)
    
    #remove any rows with whitespace
    data = data[(data['Phrase'] != "") & (data['Phrase'] != " ")]
    data = data.sample(frac=1).reset_index(drop=True)
    
    X, y = data['Phrase'].values, to_categorical(data['sentiment'].values)
    return X, y


#######Tokenize the text

def get_tokens(text, max_words):
    tokenizer = Tokenizer(num_words=num_words, lower=False)
    tokenizer.fit_on_texts(text)
    X = tokenizer.texts_to_sequences(text)
    X = pad_sequences(X, maxlen=max_words)
    
    return X


######Create the Neural network

def create_model(num_words, max_words):
    model=Sequential()
    
    #Add Input Embedding Layer
    model.add(Embedding(num_words, 128, input_length=max_words))
    
    #Add Hidden layers 
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(100, dropout=0.5, recurrent_dropout=0.5)))
    model.add(Dropout(0.5))
    
    #Add Output Layer
    model.add(Dense(2, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

#######Split the data into train, validation and test set

X_train, X_val, y_train, y_val = train_test_split(X[:153288],y[:153288], test_size = 0.2, random_state = 42)

#the rest of the data (20%) constitute the test set
X_test, y_test = X[153288:], y[153288:]

#########Train and test the model

#train
model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=64, epochs=20)

#test
model.evaluate(X_test, y_test, verbose=2, batch_size=32)

######load and process the twitter data

def prepare_tweets(path):
    data = pd.read_csv(path)
    tweets = data['Tweet'].values
    
    #convert all objects of the column to strings
    tweets = [str(x) for x in tweets]
    
    return tweets

predictor = get_tokens(tweets, max_words)

######Run model
sentiment = model.predict_classes(predictor)