#!/usr/bin/env python
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
from nltk.corpus import wordnet as wn
import scipy as sci
import numpy as np
DEBUG = 0 #CONTROL DEBUG MODE

#word_pair_score1: human judged
#word_pair_score2: machine judged

def pearson_co(word_pair_score1, word_pair_score2):
    result = 0.0
    w1_score = []
    w2_score = []
    for pair1 in word_pair_score1:
        for pair2 in word_pair_score2:
            if ( (pair1[0] == pair2[0]) and (pair1[1] == pair2[1]) ):
                #the w1,w2 and w1*, w2* are same.
                w1_score.append( float(pair1[2]) )
                w2_score.append( pair2[2] )
                break
    result = sci.stats.pearsonr(w1_score, w2_score)
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
    for w in word_vec:
        print "w0",w[0]
        print "w1",w[1]
        print "w2",w[2]
#    word_vec = read_word_vectors("./test_vector/100_3.vec")
