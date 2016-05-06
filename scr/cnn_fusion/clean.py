#!/usr/bin/env python
from nltk.corpus import stopwords
import re
"""
This file is used to clean corpus by following steps:
    1. replace all non-letter symbols (i.e.: numbers, punctations) by a space, " ".
    2. make all words to lowercase.
    3. remove all stop words.
Create Date: 7th Jan 2016
"""

DEBUG = 0
def clean_file( file_name ):
#input progress
    target_file = open(file_name, 'r')
    corpus_origin = target_file.read()

    #1. Remove all non-letter symbols.

    corpus_origin = re.sub("[^a-zA-Z]"," ", corpus_origin)
     
    #2. Make all words to lowercase.
    
    corpus_words = corpus_origin.lower().split()

    #3. Remove stop words
    stops = set(stopwords.words("english"))
    corpus_words = [w for w in corpus_words if not w in stops]

    corpus_filtered = (" ".join(corpus_words))

    target_file.close()
#output progress
    output_file_name = file_name + OUTPUT_FILE_POSTFIX
    output_file = open(output_file_name, "w")
    output_file.write(corpus_filtered)
    output_file.close()
if __name__ == "__main__":
    file_name = "wiki151201"
    clean_file(file_name)
