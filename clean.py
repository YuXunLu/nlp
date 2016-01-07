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
FILE_PREFIX = "wiki_"
FILE_COUNTS = 72
DEBUG = 1
def clean_file( file_name ):
#input progress
    target_file = open(file_name, 'r')
    corpus_origin = target_file.read()
    #0. Remove <doc id=.....> and </doc> marks
    corpus_origin = re.sub("\b<doc.*\b>]", " ", corpus_origin)
    corpus_origin = re.sub("\b</doc>]", " ", corpus_origin)
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
    output_file_name = file_name + ".clean"
    output_file = open(output_file_name, "w")
    output_file.write(corpus_filtered)
    output_file.close()
if __name__ == "__main__":
    if DEBUG:
        FILE_COUNTS = 1
    for i in range(0, FILE_COUNTS + 1):
        if i < 10:
            file_name = FILE_PREFIX + "0" + str(i)
        else:
            file_name = FILE_PREFIX + str(i)
        if DEBUG:
            print file_name
        clean_file(file_name)
