#Simple-Linear Version of Sense Fusion
import nlp_lib as nlp
import numpy as np
import scipy as sci

CSV_DIR = "../../csv/"
CSV_NAME = "M&C-30.csv"
VECTOR_DIR = "../test_vector/"
VECTOR_NAME = "100_8.vec"
VECTOR_DIM = 100
L_RATE = 0.5
epsilon = 1e-11
word_hypernyms = {}
word_hyponyms = {}
word_synonyms = {}
word_vectors = {}
word_pairs = {}
word_pool = {}

senses = {}
sense_vectors = {}
sense_hyponyms = {}
sense_hypernyms = {}
sense_synonyms = {}
sense_vectors = {}
vocab = []
def cos_function(x,y):
    result = 0
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    top = np.dot(x, np.transpose(y) )
    result = norm_x * norm_y
    result = top / result
    return result
def test_sense_vectors():
    human_score = []
    machine_score = []
    for p in word_pairs:
        w1 = p[0]
        w2 = p[1]
        human_score.append( float(p[2]) )
        m_score = cos_function( word_vectors[w1], word_vectors[w2] )
        machine_score.append(m_score)
    p_val, p_rel = sci.stats.spearmanr(human_score, machine_score)
    print "Sense-Agnostic approach:", p_val
    human_score = []
    machine_score = []
    for p in word_pairs:
        w1 = p[0]
        w2 = p[1]
        human_score.append( float(p[2]))
        m_score = cos_function(word_pool[w1], word_pool[w2])
        machine_score.append(m_score)
    p_val, p_rel = sci.stats.spearmanr(human_score, machine_score)
    print "Sense-Agnostic Pooling Approach:", p_val
    machine_score = []
    human_score = []
    for p in word_pairs:
        w1 = p[0]
        w2 = p[1]
        m_score = 0.0
        for s1 in senses[w1]:
            for s2 in senses[w2]:
                if (sense_vectors.has_key(s1) and sense_vectors.has_key(s2)):
                    m_score = m_score + cos_function(sense_vectors[s1] , sense_vectors[s2])
        m_score = m_score / ( float( len(senses[w1] ) ) * float ( len(senses[w2])) )
        if ( m_score > 0.0 ):
            machine_score.append(m_score)
            human_score.append(float(p[2]))
    p_val, p_rel = sci.stats.spearmanr(human_score, machine_score)
    print "Simple Linear Approach", p_val
if __name__ == "__main__":
    word_vectors = nlp.read_word_vectors(VECTOR_DIR + VECTOR_NAME)
    word_pairs = nlp.read_csv(CSV_DIR + CSV_NAME)
    vocab = []
    for p in word_pairs:
        vocab.append(p[0])
        vocab.append(p[1])
    vocab = list(set(vocab))
    for w in vocab:
        word_hypernyms[w] = nlp.read_hypernyms(w)
        word_hyponyms[w] = nlp.read_hyponyms(w)
        word_synonyms[w] = nlp.read_synonyms(w)
        senses[w] = nlp.read_senses(w)
        for s in senses[w]:
            sense_vectors[s] = np.zeros(VECTOR_DIM)
            sense_hypernyms[s] = nlp.read_hypernyms_by_sense(s)
            sense_hyponyms[s] = nlp.read_hyponyms_by_sense(s)
            sense_synonyms[s] = nlp.read_synonyms_by_sense(s)
            sense_vectors[s] = nlp.get_pooling(s, sense_hypernyms,sense_synonyms,sense_hyponyms, word_vectors, VECTOR_DIM)
            if ( word_vectors.has_key(w)):
                sense_vectors[s] = sense_vectors[s] + word_vectors[w]
        word_pool[w] = nlp.get_pooling(w, word_hypernyms, word_synonyms, word_hyponyms, word_vectors, VECTOR_DIM)
    test_sense_vectors()
