# nlp
For models and corpus, and some experimental files stored here.
STEPS:
0. Remove <doc id=....> and </doc> marks in documents.
1. Remove all punctations and numbers. In a short, all non-letters.
2. Convert all words to lowercase.
3. Remove Stop words.

CORPUS:
1. splited_cleaned_corpus.tar.gz contains all corpus followed by all 3 steps. It contains wiki_NUM.cln.clean, NUM from [0,10]
2. splited_original_corpus_wiki151201.tar.gz contains all original corpus directly extracted from wikipedia's dump.
3. wiki151201 contains all documents in (2), emerged into one file. Compressed file for it is wiki151201.tar.gz
4. wiki151201.clean contains all documents in (1), emerged into one file. Compressed file for it is wiki151201.clean.tar.gz
