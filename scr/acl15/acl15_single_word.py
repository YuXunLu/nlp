CSV_DIR = "../../csv/"
VECTOR_DIR = "../test_vector/"
VECTOR_NAME = "100_3.vec"
VECTOR_DIM = 100
TEST_WORD = "bank"
UPDATE_THRESHOLD = 0.01

word_vectors = {}
word_hypernyms = {}
word_hyponyms = {}
word_synonyms = {}
word_senses = {}

sense_hypernyms = {}
sense_hyponyms = {}
sense_synonyms = {}
sense_vectors = {}

alpha = 1.0
beta_synonyms = 1.0
beta_hypernyms = 0.5
beta_hyponyms = 0.5
import nlp_lib as nlp
import numpy as np
def update_sense_vector(word, sense):
    top = alpha * word_vectors[word]
    bottom = alpha
    sum_synonym = np.zeros( VECTOR_DIM )
    sum_hyponym = np.zeros( VECTOR_DIM )
#1st: update via hyponyms
    for hypon in sense_hyponyms[sense]:
        if ( word_vectors.has_key(hypon) ):
            top = top + beta_hyponyms * word_vectors[hypon]
            bottom = bottom + beta_hyponyms
#2nd: update via hypernyms
    for hyper in sense_hypernyms[sense]:
        if ( word_vectors.has_key(hyper) ):
            top = top + beta_hypernyms * word_vectors[hyper]
            bottom = bottom + beta_hypernyms
#3nd: update via synonyms
    for syn in sense_synonyms[sense]:
        if ( word_vectors.has_key(syn) ):
            top = top + beta_synonyms * word_vectors[syn]
            bottom = bottom + beta_synonyms

    final_vector = top / bottom

    return final_vector
if __name__ == "__main__":
    word_vectors = nlp.read_word_vectors(VECTOR_DIR + VECTOR_NAME)
    word_senses[TEST_WORD] = nlp.read_senses(TEST_WORD)
#initialize
    for s in word_senses[TEST_WORD]:
        sense_vectors[s] = word_vectors[TEST_WORD]
        sense_hypernyms[s] = nlp.read_hypernyms_by_sense(s)
        sense_hyponyms[s] = nlp.read_hyponyms_by_sense(s)
        sense_synonyms[s] = nlp.read_synonyms_by_sense(s)
#update stage
    t = 0 
    for s in word_senses[TEST_WORD]:
        current_s_vector = np.zeros(VECTOR_DIM)
        error = 10000000.0
        iter = 1
        while ( error > UPDATE_THRESHOLD ):
            print "[current sense]", s
            print "[iteration]",iter
            print "[error]",error
            current_s_vector = update_sense_vector(TEST_WORD, s)
            error = np.linalg.norm( current_s_vector - sense_vectors[s] )
            iter = iter + 1
            if (iter > 10):
                break
        sense_vectors[s] = current_s_vector
#compute distance between each senses
    for s in word_senses[TEST_WORD]:
        for s1 in word_senses[TEST_WORD]:
            print "Sense 1",s, "Sense 2" , s1
            print "Their distance", np.linalg.norm ( sense_vectors[s] - sense_vectors[s1] )



