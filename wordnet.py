import nltk

from nltk.corpus import wordnet

#Returns an array of synonmes sets
syn_example = wordnet.synsets("program")

print(syn_example)
print(syn_example[0].lemmas()[0].name())

bad_synonyms = []
bad_set = wordnet.synsets("bad")

for synset in bad_set:
    bad_synonyms.append(synset.lemmas()[0].name())

#Set removes the repeated synonym lemmas
print(set(bad_synonyms))