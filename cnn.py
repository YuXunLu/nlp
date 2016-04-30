from nltk.corpus import wordnet as wn
import nlp_lib as nlplib
import numpy as np
word_vectors = {}
new_word_vectors = {}
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
def cnn_calc(w1,w2):
    #Index for hypernyms, synonyms, hyponyms, respectively.
    i = 0
    j = 0
    k = 0
    feature_map = []
    #STEP 1: BUILDING FEATURE MAP, width = 3
    print "Build feature map"
    if (word_hypernyms.has_key(w1) and word_synonyms.has_key(w1) and word_hyponyms.has_key(w1) ):
        while ( i < len(word_hypernyms[w1]) and j < len(word_synonyms[w1]) and k < len(word_hyponyms[w1] )):
            w1_hypernym = word_hypernyms[w1][i]
            w1_synonym = word_synonyms[w1][j]
            w1_hyponym = word_hyponyms[w1][k]            
            #must find a hypernym
            while( not (word_vectors.has_key(w1_hypernym) and i < len(word_hypernyms[w1]) ) ):
                i = i + 1
                w1_hypernym = word_hypernyms[w1][i]
            #if we didn't find any hypernym
            if ( i > len(word_hypernyms) ):
                break
            else: #we got a hypernym
                feature_map.append( word_vectors[w1_hypernym] )
                i = i + 1
            
            #must find a synonym
            while( not (word_vectors.has_key(w1_synonym) and j < len(word_synonyms[w1]) ) ):
                j = j + 1
                w1_synonym = word_synonyms[w1][j]
            if ( j > len(word_synonyms) ):
                berak;
            else:
                feature_map.append( word_vectors[w1_synonym] )
                j = j + 1

            #must find a hyponym
            while( not(word_vectors.has_key(w1_hyponym) and k < len(word_hyponyms[w1]) )):
                k = k + 1
                w1_hyponym = word_hyponyms[w1][k]
            if ( k > len(word_hyponyms) ):
                break;
            else:
                feature_map.append (word_vectors[w1_hyponym] )
                k = k + 1
    #STEP 2: CONVOLUTION STAGE
    print "convolution"
    if ( len(feature_map) >= 3 ):
        #at least 3 nodes in feature map, which correspondent to a convolutional layer node
        conv_nodes_num = len(feature_map) - 2 # convolution nodes number
        convolution_result = []
        i = 0
        while ( i < conv_nodes_num ):
            node_feature = np.zeros( (3, word_vector_dim) , np.float64)
            node_feature[0] = feature_map[i]
            node_feature[1] = feature_map[i+1]
            node_feature[2] = feature_map[i+2]
            #convolution
            ADD_VEC = np.ones( (1,3), np.float64 )
            #shape
            print "shape ADD_VEC", ADD_VEC.shape()
            print "shape weight conv", weight_convolution.shape()
            print "node_feature", node_feature.shape()
            print "bias_vector", bias_vector.shape()
            this_node_result = np.add( np.dot(ADD_VEC, np.dot(weight_convolution, node_feature)) , bias_vector)
            convolution_result.append(this_node_result)
            i = i + 1
    else:
        print "warning, convolution nodes insufficient","[word]:", w1
        return np.zeros( (3, word_vector_dim), np.float64 )
    #STEP 3: POOLING
    print "pooling"
    final_vector = np.zeros( (3, word_vector_dim), np.float64 )
    conv_nodes_N = np.array(conv_nodes_num, np.float64)
    for v in convolution_result:
        final_vector = np.add(final_vector, v)
    final_vector = np.divide(final_vector, conv_nodes_N)
    #return v_{x}^*, and feature map for weight updating.
    return final_vector, feature_map
def cnn_training():
    #initialize weight matrix and bias vector there!
    weight_matrix = np.identity( 3, np.float64 )
    bias_vector = np.zeros( (1, word_vector_dim), np.float64)
    print "Training."
    for word_pairs in word_pair_score:
        word_1 = word_pairs[0]
        word_2 = word_pairs[1]
        score = np.float32(float(word_pairs[2]))
        print "word1",word_1
        print "vector"
        print cnn_calc(word_1,word_2)
if __name__=="__main__":
    print "read vector & score"
    word_vectors = nlplib.read_word_vectors("./test_vector/100_3.vec")
    word_pair_score = nlplib.read_csv("./csv/R&G-65.csv")
######Read hypernyms
    print "get hypernyms etc."
    for w_pair_score in word_pair_score:
        word_dictionary.append(w_pair_score[0])
        word_dictionary.append(w_pair_score[1])
#   remove dumplicated word, for we are searching hypernyms, hyponyms, synonyms according to the dictionary.
    word_dictionary = list(set(word_dictionary))
    for w in word_dictionary:
        word_hypernyms[w] = nlplib.read_hypernyms(w)
        word_hyponyms[w] = nlplib.read_hyponyms(w)
        word_synonyms[w] = nlplib.read_synonyms(w)
    print "start training"
    cnn_training()
