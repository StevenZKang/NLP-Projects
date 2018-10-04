import nltk
import random

from nltk.corpus import movie_reviews

reviews = [(list(movie_reviews.words(fileid)), category)
           for category in movie_reviews.categories()
           for fileid in movie_reviews.fileids(category)]

#Shuffle Reviews to seperate training and testing data.
random.shuffle(reviews)
#print(reviews[1])

#Store all review words into an array
all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

#Translate the array of words into a Frequency Dict
all_words = nltk.FreqDist(all_words)

#Distribution of 3000 most common words in reviews
word_features = list(all_words.keys())[:3000]

def find_features(reviews: list)->dict:
    """Takes one review and searches for the 3000 words in word features,
    returning a dictionary marking whether they are present or not"""
    words = set(reviews)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

feature_presence = find_features(movie_reviews.words('neg/cv000_29416.txt'))

featuresets = [(find_features(rev), category) for (rev, category) in reviews]

print(featuresets)