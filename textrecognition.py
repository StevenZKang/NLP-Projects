import nltk

from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

def tag_text(token_text: list) -> list:
    "Tags the input text in tuples"
    final_list = []
    for sentence in token_text:
        try:
            words = nltk.word_tokenize(sentence) #Tokenize each word in the sentence into a list
            tagged = nltk.pos_tag(words) #Tag the individual word depending on part of speech
            final_list.append(tagged) #Append the list of tuples to final list


        except Exception as e:
            print(str(e))
    return final_list

def chunk_text(tagged_text: tuple):
    "Takes a list of tagged tuples and returns a noun chunk"
    chunkGram = r"""Chunk: {<RB.?>*<VB?>*<NNP>+<NN>?}"""
    chunkParser = nltk.RegexpParser(chunkGram)
    chunked = chunkParser.parse(tagged_text)
    return chunked

training_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

custom_sent_tokenizer = PunktSentenceTokenizer(training_text)

#Custom Sentence Tokenizing
tokenized_text = custom_sent_tokenizer.tokenize(sample_text)

tagged_text = tag_text(tokenized_text)
print(tagged_text)
chunked_text = chunk_text(tagged_text[0])
chunked_text.draw()




