import nltk
import random
import pickle
import preprocessing
import textrecognition

from nltk.corpus import movie_reviews

def find_features(review: list, word_features: list)->dict:
    """Takes one review and searches for the 3000 words in word features,
    returning a dictionary marking whether they are present or not"""
    words = set(review)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

feature_presence = find_features(movie_reviews.words('neg/cv000_29416.txt'))

def open_classifer():
    #Open the saved byte file with the saved classifer and load it into classifier
    classifier_r = open("naivebayes,pickle", "rb")
    classifier = pickle.load(classifier_r)
    #Close the byte file
    classifier_r.close()
    return classifier

def save_classifier(classifier: object):
    # Save the classifier into file.
    save_classifier = open("naivebayes.pickle", "wb")
    pickle.dump(classifier, save_classifier)
    save_classifier.close()

def create_word_features(top_n: int):
    """Distributes all words in imported movie reviews to return a list
    of the most common top_n words"""

def create_feature_sets():
    """"""

def standard_training():
    """
    :return:
    """

if __name__== "__main__":

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
    #(rev,category) = (review, pos/neg)
    #Creates a list of tuples where the first element is a dict of the presence of the words
    #and the second element is pos/neg.
    feature_sets = [(find_features(review, word_features), rating) for (review, rating) in reviews]

    #Split Training and Testing Data
    training_set = feature_sets[:1900]
    testing_set = feature_sets[1900:]

    #Declare classifier algorithm
    classifier = nltk.NaiveBayesClassifier.train(training_set)

    #Nltk.classify.accuracy takes in a testing set and runs in through the given classifier to determine accuracy.
    print("Stage One Results")
    print("Accuracy Percentage:", (nltk.classify.accuracy(classifier,testing_set))*100)

    classifier.show_most_informative_features(10)