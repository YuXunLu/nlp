from nltk.corpus import wordnet as wn
import nlp_lib as nlplib
import numpy as np
word_vectors = {}
word_vector_dim = 100
word_pair_score = []
word_dictionary = []
word_hypernyms = {}
word_hyponyms = {}
word_synonyms = {}
#CNN part
weight_convolution = []
bias_vector = []
def lost_function(vx,vy,score):
    return 0.0
def cnn_calc(w1,w2,score):
    #Index for hypernyms, synonyms, hyponyms, respectively.
    i = 0
    j = 0
    k = 0
    s = score
    feature_map = []
    #STEP 1: BUILDING FEATURE MAP, width = 3
    if (word_hypernyms.has_key(w1) && word_synonyms.has_key(w1) && word_hyponyms.has_key(w1) ):
        while ( i < len(word_hypernyms[w1]) and j < len(word_synonyms[w1]) and k < len(word_hyponyms[w1] )):
            w1_hypernym = word_hypernyms[w1][i]
            w1_synonym = word_synonyms[w1][j]
            w1_hyponym = word_hyponyms[w1][k]
            feature_node = np.array( zeros(3,word_vector_dim) )
            if ( word_vectors.has_key(w1_hypernym) ):
                feature_node[1] = word_vectors[w1_hypernym]
                if (word_vectors.has_key(w1_synonym) ):
                    feature_node[2] = word_vectors[w1_synonym]
                    if ( word_vectors.has_key(w1_hyponym) ):
                        feature_node[3] = word_vectors[w1_hyponym]
                        feature_map.add(feature_node)
                        feature_node = np.array( zeros(3,word_vector_dim) )
                        i = i + 1
                        j = j + 1
                        k = k + 1
                    else:
                        k = k + 1
                else:
                    j = j + 1
            else:
                i = i + 1
    #STEP 2: CONVOLUTION STAGE
    if ( len(feature_map) >= 3 ):
        #at least 3 nodes in feature map, which correspondent to a convolutional layer node


def cnn_training():
    for word_pairs in word_pair_score:
        word_1 = word_pairs[0]
        word_2 = word_pairs[1]
        score = np.float32(float(word_pairs[3]))
        cnn_calc(word_1,word_2,score)
if __name__=="__main__":
    word_vectors = nlplib.read_word_vectors("./test_vector/100_3.vec")
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
