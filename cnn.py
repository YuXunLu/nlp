from nltk.corpus import wordnet as wn
import nlp_lib as nlplib
import numpy as np
word_vector = {}
word_pair_score = []
word_dictionary = []
word_hypernyms = {}
word_hyponyms = {}
word_synonyms = {}
#CNN part
weight_convolution = []
def lost_function(vx,vy,score):
    return 0.0
def cnn_calc(w1,w2,score):
    i = 0
    s = score
    while ( ( i < len(word_hypernyms[w1]) ) and ( i < len(word_hyponyms[w1])) and ( i < len(word_synonyms[w1])) ):
        #Build feature map
        node_feature = 
        i = i + 1
def cnn_training():
    for word_pairs in word_pair_score:
        word_1 = word_pairs[0]
        word_2 = word_pairs[1]
        score = np.float32(word_pairs[3])
        cnn_calc(word_1,word_2,score)
if __name__=="__main__":
    word_vector = nlplib.read_word_vectors("./test_vector/100_3.vec")
    word_pair_score = nlplib.read_csv("./csv/R&G-65.csv")
######Read hypernyms
    for w_pair_score in word_pair_score:
        word_dictionary.append(w_pair_score[0])
        word_dictionary.append(w_pair_score[1])
#   remove dumplicated word, for we are searching hypernyms, hyponyms, synonyms according to the dictionary.
    word_dictionary = list(set(word_dictionary))
    for w in word_dictionary:
        word_hypernyms[w] = nlplib.read_hypernyms(w)
        word_hyponyms[w] = nlplib.read_hyponyms(w)
        word_synonyms[w] = nlplib.read_synonyms(w)
    cnn_training()
