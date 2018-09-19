import nltk


# Lemmatization is an organized & step by step procedure of obtaining the root form of the word,
# it makes use of vocabulary (dictionary importance of words) and
# morphological analysis (word structure and grammar relations)
from nltk.stem.wordnet import WordNetLemmatizer
lem = WordNetLemmatizer()

#Stemming is a rudimentary rule-based process of stripping the suffixes (“ing”, “ly”, “es”, “s” etc) from a word.
from nltk.stem.porter import PorterStemmer
stem = PorterStemmer()

word = "multiplying"
print(lem.lemmatize(word,"v"))

print(stem.stem(word))

#Tokenizer
from nltk.tokenize import sent_tokenize, word_tokenize
sample_text = "Welcome to Toronto, this is Canada's greatest city!"
sample_tokens = word_tokenize(sample_text)

#Stopwords
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

filtered_sentence = [x for x in sample_tokens if x not in stop_words]
print(filtered_sentence)

