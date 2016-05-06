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
word_senses_hyponyms = {}

word_hyponyms = {}
word_hypernyms = {}
word_vectors = {}
word_synonyms = {}

word_final_vector = {}
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
            word_senses_hyponyms[s] = nlp_lib.read_hyponyms_by_sense(s)
    #read for retrofitting
    for w in vocab:
        word_hyponyms[w] = nlp_lib.read_hyponyms(w)
        word_hypernyms[w] = nlp_lib.read_hypernyms(w)
        word_synonyms[w] = nlp_lib.read_synonyms(w)
    #read word vectors

    word_vectors = nlp_lib.read_word_vectors(VECTOR_DIR + VECTOR_NAME)
