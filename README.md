# nlp
CBOW: simple.py

Sense Pooling: simple_linear.py

Optimized Sense Pooling: opt1_linear_new.py

Neural Network Model: NN_hidden.py

Word Vectors used in this model could be download at:



The Result of NN model with C(s_i)^{(t-1)} - C_(s_i)^{(t)} > epsilon (= 1e-8):







For models and corpus, and some experimental files stored here.
STEPS:
0. Remove <doc id=....> and </doc> marks in documents.
1. Remove all punctations and numbers. In a short, all non-letters.
2. Convert all words to lowercase.
3. Remove Stop words.

CORPUS
1. splited_original_corpus_wiki151201.tar.gz contains all original corpus directly extracted from wikipedia's dump.
2. wiki151201 contains all documents in (2), emerged into one file. Compressed file for it is wiki151201.tar.gz

DEVELOPMENT NOTES

H28 May 28th,:
Project reopen for my essay on NAIST.

sed "\<doc\d;\doc>\d" wiki151201 - command used to clean wiki151201

there are 48 different words in R&G-65
