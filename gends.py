# Generates the dataset and saves it to 'dataset/movie_dataset.npz'.

import os
import pickle
import numpy as np
import pandas as pd
from utils import nlp_ize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelBinarizer
from scipy import sparse

url = 'https://raw.githubusercontent.com/meghjoshii/NSDC_DataScienceProjects_SentimentAnalysis/main/IMDB%20Dataset.csv'

# Read the data from csv fetched from the URL.

required = [ 'dataset/ds_review.npz', 'dataset/ds_sentiment.npy', 'dataset/feature_names.pkl' ]

data = None

for item in required: 
    if not os.path.exists(item):
        print('Fetching movie review DB...')

        data = pd.read_csv(url)

        break 

# Check whether the review dataset exists.

if not os.path.exists('dataset/ds_review.npz') or not os.path.exists('dataset/feature_names.pkl'):
    print('Generating review dataset:')
    ## NLP it to send it onto the model.

    print('Applying NLP...')
    data.review = data.review.apply(nlp_ize)

    # Vectorize.

    print('Appying vectorizer...')
    cv = CountVectorizer(min_df = 0.0, max_df = 1.0, binary = False, ngram_range = (1, 3))

    cv_train_data = cv.fit_transform(data.review)

    sparse.save_npz('dataset/ds_review.npz', cv_train_data)
    
    with open('dataset/feature_names.pkl', 'wb') as file: 
        pickle.dump(cv.get_feature_names_out(), file)

    print("Saved dataset to 'dataset/ds_review.npz'!")
    print("Dumped feature names to 'dataset/feature_names.pkl'!")
else: 
    print("Found review dataset in 'dataset/ds_review.npz' and feature names 'dataset/feature_names.pkl'. Nothing to do.")

if not os.path.exists('dataset/ds_sentiment.npy'):
    print('Generating sentiment dataset:')
    print('Applying binarizer...')

    lb = LabelBinarizer()

    lb_sentiment_data = lb.fit_transform(data.sentiment)

    np.save('dataset/ds_sentiment.npy', lb_sentiment_data)

    print("Saved dataset to 'dataset/ds_sentiment.npy'!")
else: 
    print("Found sentiment dataset in 'dataset/ds_sentiment.npy'. Nothing to do.")
