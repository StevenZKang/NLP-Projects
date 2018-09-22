import nltk

from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

def tag_text(token_text: list):
    "Tags the input text in tuples"
    for sentence in token_text:
        try:
            word = nltk.word_tokenize(sentence)
            tagged = nltk.pos_tag(word)
            print(tagged)

        except Exception as e:
            print(str(e))

training_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

custom_sent_tokenizer = PunktSentenceTokenizer(training_text)

#Custom Sentence Tokenizing
tokenized_text = custom_sent_tokenizer.tokenize(sample_text)

tag_text(tokenized_text)




