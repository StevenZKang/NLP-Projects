import nltk
import random

from nltk.corpus import movie_reviews

reviews = [(list(movie_reviews.words(fileid)), category)
           for category in movie_reviews.categories()
           for fileid in movie_reviews.fileids(category)]

#Shuffle Reviews to seperate training and testing data.
random.shuffle(reviews)
print(reviews[1])

#Store all review words into an array
all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

#Translate the array of words into a Frequency Dict
all_words = nltk.FreqDist(all_words)
