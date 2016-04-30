#!/usr/bin/env python
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
from nltk.corpus import wordnet as wn
DEBUG = 1 #CONTROL DEBUG MODE
def read_csv(filename, split_tag = ","):
    word_csv = []
    csv_file = open(filename,"r")
    for line in csv_file:
        words = line.rstrip().split(split_tag)
        word_csv.append(words)
    return word_csv
def read_word_vectors(filename, split_tag = " "):
    word_vector = {}
    vec_file = open(filename, "r")
    for line in vec_file:
        vec = line.rstrip().split(split_tag)
        word_text = vec[0]
        word_vector[word_text] = vec[1:]
        if DEBUG:
            print "[WORD]:", word_text
            print "[WORD_VECTOR]:", word_vector[word_text]
            print "[WORD_VEC_LENGTH]:", len(word_vector[word_text])
    map(float,word_vector[word_text])
    word_vecotr[word_text] = np.array(word_vector[word_text], float64)
    return word_vector
def read_synonyms(word):
    result = []
    word_synsets = wn.synsets(word)
    for w in word_synsets:
        for lemma in w.lemmas():
            result.append(lemma.name())
    return result
def read_hypernyms(word):
    result = []
    word_synsets = wn.synsets(word)
    for w in word_synsets:
        for hyper in w.hypernyms():
            for l in hyper.lemmas():
                result.append(l.name())
    return result
def read_hyponyms(word):
    result = []
    word_synsets = wn.synsets(word)
    for w in word_synsets:
        for hypon in w.hyponyms():
            for l in hypon.lemmas():
                result.append(l.name())
    return result
#Testing part, plz do not care about it
test_synonym = {}
if __name__ == "__main__":
#This part should never be reached unless u are in debug mode.
    word_vec = read_csv("./csv/R&G-65.csv")
    print word_vec
