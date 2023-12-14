# The REPL where you write down your review and get the sentiment.

import os
import pickle
import numpy as np
from scipy import sparse
from utils import nlp_ize
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.naive_bayes import MultinomialNB

# Check if the datasets are OK.

dataset = [ 'dataset/ds_review.npz', 'dataset/ds_sentiment.npy', 'dataset/feature_names.pkl' ]

for item in dataset: 
    if not os.path.exists(item): 
        print("Dataset not found! Try running 'python gends.py' to generate the datasets.")

        exit(-1)

train_review    = sparse.load_npz('dataset/ds_review.npz')
train_sentiment = np.load('dataset/ds_sentiment.npy')

# Get the feature names (vocabularies).

with open('dataset/feature_names.pkl', 'rb') as file: 
    feature_names = pickle.load(file)

mnb = MultinomialNB()

# Train the Bayes.

print('Training Naive Bayes...')
mnb.fit(train_review, train_sentiment.ravel())

print('Initializing vectorizer...')
cv = CountVectorizer(vocabulary = feature_names)

print('Preparing binarizer...')

lb = LabelBinarizer()

# We only have to deal with 'positive' and 'negative' words.

lb.fit([ 'positive', 'negative' ])

print('Entering REPL...', end = '\n\n')

while True: 
    print('>>> ', end = '')
    string = input()

    if string == '.exit':
        break

    string = cv.transform([nlp_ize(string)])

    response = lb.inverse_transform(mnb.predict(string))

    if response[0] == 'positive': 
        print('-> Your review is positive! Good job.')
    else:
        print('-> Your review is negative! Good luck with that.')
