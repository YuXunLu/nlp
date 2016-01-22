#!/usr/bin/env python
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
from nltk.corpus import wordnet as wn
DEBUG = 1 #CONTROL DEBUG MODE
def read_word_vectors(filename, split_tag = " "):
    word_vector = {}
    vec_file = open(filename, "r")
    for line in vec_file:
        vec = line.split(split_tag)
        word_text = vec[0]
        word_vector[ word_text ] = vec[1:]
        if DEBUG:
            print "[WORD]:", word_text
            print "[WORD_VECTOR]:", word_vector[word_text]
            print "[WORD_VEC_LENGTH]:", len(word_vector[word_text])
    return word_vector
    #Testing part, plz do not care about it
if __name__ == "__main__":
#THIS FUNCTION SHALL NEVER BE ENTERED UNLESS YOU ARE IN DEBUGGING MODE
    if DEBUG:
        vector_file_name = "./test_vector/100_3.vec"
    word_v = read_word_vectors(vector_file_name)
    print word_v
