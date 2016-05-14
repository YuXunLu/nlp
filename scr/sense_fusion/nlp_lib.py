#!/usr/bin/env python
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
from nltk.corpus import wordnet as wn
import scipy as sci
import numpy as np
DEBUG = 0 #CONTROL DEBUG MODE
def get_single_pooling(word, rel_words, word_vectors, word_dim = 100):
    i = 1.0
    result = np.zeros(word_dim)
    for w in rel_words[word]:
        if ( word_vectors.has_key(w) ):
            result = result + word_vectors[w]
            i = i + 1.0
    if ( i - 1.0 > 0.0):
        result = result / i
    return result
def get_pooling(word, word_hypernyms, word_synonyms, word_hyponyms, word_vectors, word_dim = 100):
    result = np.zeros(word_dim)
    if word_vectors.has_key(word):
        result = result + word_vectors[word]:

    hyper_pool = get_word_single_pooling(word, word_hypernyms, word_vectors, word_dim)
    syn_pool = get_word_single_pooling(word, word_synonyms, word_vectors, word_dim)
    hypon_pool = get_word_single_pooling(word, word_hyponyms, word_vectors, word_dim)

    result = result + hyper_pool + syn_pool + hypon_pool
    return result
def calc_pearson(score1, score2):
    result = []
    return result
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
        map(float,word_vector[word_text])
        word_vector[word_text] = np.array(word_vector[word_text], np.float64)
        if DEBUG:
            print "[WORD]:", word_text
            print "[WORD_VECTOR]:", word_vector[word_text]
            print "[WORD_VEC_LENGTH]:", len(word_vector[word_text])
    return word_vector
def read_synonyms(word, pos = wn.NOUN):
    result = []
    word_synsets = wn.synsets(word, pos = pos)
    for w in word_synsets:
        for lemma in w.lemmas():
            result.append(lemma.name())
    return result
def read_synonyms_by_sense(sense):
    result = []
    for lemma in sense.lemmas():
        result.append(lemma.name())
    return result
def read_hypernyms(word, pos = wn.NOUN):
    result = []
    word_synsets = wn.synsets(word, pos = pos)
    for w in word_synsets:
        for hyper in w.hypernyms():
            for l in hyper.lemmas():
                result.append(l.name())
    return result
def read_hypernyms_by_sense(sense):
    result = []
    for hyper in sense.hypernyms():
        for l in hyper.lemmas():
            result.append(l.name())
    return result
def read_hyponyms(word, pos = wn.NOUN):
    result = []
    word_synsets = wn.synsets(word, pos = pos)
    for w in word_synsets:
        for hypon in w.hyponyms():
            for l in hypon.lemmas():
                result.append(l.name())
    return result
def read_hyponyms_by_sense(sense):
    result = []
    for hypon in sense.hyponyms():
        for l in hypon.lemmas():
            result.append(l.name())
    return result
def read_senses(word, pos = wn.NOUN):
    result = []
    w_synsets = wn.synsets(word, pos = pos)
    for s in w_synsets:
        s_lemmas = s.lemmas()
        l_name = [str(l.name()) for l in s_lemmas]
        if DEBUG:
            print "synset:",s
            print "lemma name:",l_name
        if (word in l_name):
            result.append(s)
    return result
test_synonym = {}
if __name__ == "__main__":
#This part should never be reached unless u are in debug mode.
    bank_senses = read_senses("bank")
    print bank_senses
    
