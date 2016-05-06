from nltk.corpus import wordnet as wn
import nlp_lib as nlp_lib
import numpy as np
import scipy as sci
VECTOR_DIR ="../test_vector/"
VECTOR_NAME = "100_3.vec"
CSV_DIR = "../../csv/"
CSV_NAME = "R&G-65.csv"
VECTOR_DIM = 100

vocab = []
word_pairs = []
word_senses = {}
word_sense_hyponyms = {}
word_sense_vectors = {}


word_hyponyms = {}
word_hypernyms = {}
word_vectors = {}
word_synonyms = {}

word_final_vectors = {}
word_hypon_pooling = {}
def cos_sim(v1, v2):
    result = 0.0
    v1_len = np.dot( np.transpose(v1), v1)
    v1_len = np.sqrt(v1_len)

    v2_len = np.dot( np.transpose(v2), v2)
    v2_len = np.sqrt(v2_len)

    up = np.dot( np.transpose(v1), v2)
    bottom = v2_len * v1_len

    result = up/bottom
    return result
def get_hypon_pooling(word):
    i = 0.0
    hypon_pool = np.zeros(VECTOR_DIM)
    for hypon in word_hyponyms[word]:
        if (word_vectors.has_key(hypon) ):
            hypon_pool = hypon_pool + word_vectors[hypon]
            i = i + 1.0
    if ( (i-1.0) > 0 ):
        hypon_pool = hypon_pool / i
    return hypon_pool
def get_pooling(word):
    
    if ( word_vectors.has_key(word) ):
        result = word_vectors[word]
    else:
        result = np.zeros(VECTOR_DIM)
    
    i = 0.0
    hyper_pool = np.zeros(VECTOR_DIM)
    for hyper in word_hypernyms[word]:
        if (word_vectors.has_key(hyper)):
            hyper_pool = hyper_pool + word_vectors[hyper]
            i = i + 1.0
    if (( i - 1.0 ) > 0.0):
        hyper_pool = hyper_pool / i
    result = result + hyper_pool

    j = 0.0
    hypon_pool = np.zeros(VECTOR_DIM)
    for hypon in word_hyponyms[word]:
        if (word_vectors.has_key(hypon)):
            hypon_pool = hypon_pool + word_vectors[hypon]
            j = j + 1.0
    if (( j -1.0 ) > 0.0 ):
        hypon_pool = hypon_pool / j
    result = result + hypon_pool

    k = 0.0
    synonym_pool = np.zeros(VECTOR_DIM)
    for synonym in word_synonyms[word]:
        if (word_vectors.has_key(synonym) ):
            synonym_pool = synonym_pool + word_vectors[synonym]
            k = k + 1.0
    if ( (k-1.0) > 0.0):
        synonym_pool = synonym_pool / k
    result = result + synonym_pool
    return result

#calculating the sense vector
def CNN_calc(sense, word, paddle, feature_map, weight_mat):
    #the convolutional node # = feature # - 2
    conv_num = len(feature_map) - 2
    node_feature = np.array( (3, VECTOR_DIM) )
    conv_result = []
    result = np.zeros(VECTOR_DIM)
    i = 0
    #convolution step
    while ( i < conv_num ):

        res = np.zeros(VECTOR_DIM)

        node_feature[0] = feature_map[i]
        node_feature[1] = feature_map[i+1]
        node_feature[2] = feature_map[i+2]     

        res = res + node_feature[0] * weight_mat[0][0] + node_feature[1] * weight_mat[1][1] + node_feature[2] * weight_mat[2][2]
        conv_result.append(res)
        i = i + 1
    #pooling step
    i = 0
    while ( i < len(conv_result) ):
        result = result + conv_result[i]
        i = i + 1
    result = result / len(conv_result)
    return result
#using cnn train sense vector
def train_CNN(sense, word):
    sense_vector = np.zeros(VECTOR_DIM)
    paddle = np.zeros(VECTOR_DIM)
    weight_mat = np.identity(3)
    feature_map = []

    feature_map.append(paddle)

    #step1: build feature map
    for hypon in word_sense_hyponyms[sense]:
        if ( word_vectors.has_key(hypon) ):
            feature_map.append(word_vectors[hypon])

    feature_map.append(paddle)

    num_feature_map = len(feature_map)

    #step 2: calc cnn
    if ( num_feature_map > 2 ):
        tmp_sense_vector = CNN_calc(sense, word, feature_map, weight_mat)
        while(tmp_sense_vector
    return sense_vector
#build and train sense vectors
def training_sense_vectors():
    for w in vocab:
        if ( len(word_senses[w]) > 0 ): #polynonmy
            for s in word_senses[w]:
                if ( len(word_sense_hyponyms[s]) > 0 ): #a sense with many hyponyms


if __name__ == "__main__":
    word_pairs = nlp_lib.read_csv( CSV_DIR + CSV_NAME )
    for w_pair in word_pairs:
        vocab.append( w_pair[0] )
        vocab.append( w_pair[1] )
    #remove duplicated words in csv file
    vocab = list(set(vocab))
    #read word senses
    for w in vocab:
        word_senses[w] = nlp_lib.read_senses(w)
    #read word senses' hyponyms
    for w in vocab:
        for s in word_senses[w]:
            word_sense_hyponyms[s] = nlp_lib.read_hyponyms_by_sense(s)
    #read for retrofitting
    for w in vocab:
        word_hyponyms[w] = nlp_lib.read_hyponyms(w)
        word_hypernyms[w] = nlp_lib.read_hypernyms(w)
        word_synonyms[w] = nlp_lib.read_synonyms(w)
    #read word vectors

    word_vectors = nlp_lib.read_word_vectors(VECTOR_DIR + VECTOR_NAME)

    for w in vocab:
        word_final_vectors[w] = get_pooling(w)
        word_hypon_pooling[w] = get_hypon_pooling(w)
    training_sense_vectors()
