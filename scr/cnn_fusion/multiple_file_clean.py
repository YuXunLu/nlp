#!/usr/bin/env python
from nltk.corpus import stopwords
import re
import multiprocessing
"""
This file is used to clean corpus by following steps:
    1. replace all non-letter symbols (i.e.: numbers, punctations) by a space, " ".
    2. make all words to lowercase.
    3. remove all stop words.
Create Date: 7th Jan 2016
"""
FILE_PREFIX = "wiki_"
INPUT_FILE_POSTFIX = ".cln"
OUTPUT_FILE_POSTFIX = ".clean"
FILE_COUNTS = 10
DEBUG = 0
def clean_file( file_name ):
#input progress
    print "[INPUT STAGE] Filename:" + file_name

    target_file = open(file_name, 'r')
    corpus_origin = target_file.read()

    print "[CLEAN STAGE] Filename:" + file_name
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
    print "[OUTPUT STAGE] Filename:" + output_file_name
    output_file = open(output_file_name, "w")
    output_file.write(corpus_filtered)
    output_file.close()
if __name__ == "__main__":
    print "Main-Process starts"
    pool = multiprocessing.Pool(processes=5)
    if DEBUG:
        FILE_COUNTS = 72
    for i in range(0, FILE_COUNTS + 1):
        if i < 10:
            file_name = FILE_PREFIX + "0" + str(i)
        else:
            file_name = FILE_PREFIX + str(i)
        file_name = file_name + INPUT_FILE_POSTFIX
        if DEBUG:
            print file_name
        pool.apply_async( clean_file, (file_name,) )
    pool.close()
    pool.join()
    print "Sub-process(es) done."
    print "Main-process done."
