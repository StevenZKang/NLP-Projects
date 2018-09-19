import nltk


# Lemmatization is an organized & step by step procedure of obtaining the root form of the word,
# it makes use of vocabulary (dictionary importance of words) and
# morphological analysis (word structure and grammar relations)
from nltk.stem.wordnet import WordNetLemmatizer
lem = WordNetLemmatizer()

#Stemming is a rudimentary rule-based process of stripping the suffixes (“ing”, “ly”, “es”, “s” etc) from a word.
from nltk.stem.porter import PorterStemmer
stem = PorterStemmer()

from nltk.tokenize import sent_tokenize, word_tokenize
sample_text = "Welcome to Toronto, this is Canada's greatest city!"
print(word_tokenize(sample_text))

word = "multiplying"
print(lem.lemmatize(word,"v"))

print(stem.stem(word))