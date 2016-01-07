#!/usr/bin/env python
"""
This file is used to clean corpus by following steps:
    1. replace all non-letter symbols (i.e.: numbers, punctations) by a space, " ".
    2. make all words to lowercase.
    3. remove all stop words.
Create Date: 7th Jan 2016
"""
FILE_PREFIX = "wiki_"
FILE_COUNTS = 72
DEBUG = 0
if __name__ == "__main__":
    for i in range(0, FILE_COUNTS):
        if i < 10:
            file_name = FILE_PREFIX + "0" + str(i)
        else:
            file_name = FILE_PREFIX + str(i)
        if DEBUG:
            print file_name

