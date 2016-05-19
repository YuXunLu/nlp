#Non-Linear Version of Sense Fusion
import nlp_lib as nlp
import numpy as np
import scipy as sci

CSV_DIR = "../../csv/"
CSV_NAME = "R&G-65.csv"
VECTOR_DIR = "../test_vector/"
VECTOR_NAME = "100_8.vec.sense"
VECTOR_DIM = 100
L_RATE = 0.5
alpha = 1.0
beta_syn = 1.0
beta_hyper = 0.5
beta_hpyon = 0.5
epsilon = 0.01
max_iter = 10

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
    machine_score = []
    human_score = []
    for p in word_pairs:
        w1 = p[0].lower()
        w2 = p[1].lower()
        m_score = 0.0
        for v1 in sense_vectors[w1]:
            for v2 in sense_vectors[w2]:
                #if (sense_vectors.has_key(s1) and sense_vectors.has_key(s2)):
                m_score = m_score + cos_function(v1 , v2)
        m_score = m_score / ( float( len(sense_vectors[w1] ) ) * float ( len(sense_vectors[w2])) )
        if ( m_score > 0.0 ):
            machine_score.append(m_score)
            human_score.append(float(p[2]))
    p_val, p_rel = sci.stats.spearmanr(human_score, machine_score)
    print "ACL15 Approach", p_val
if __name__ == "__main__":
    word_vectors = nlp.read_word_vectors(VECTOR_DIR + VECTOR_NAME)
    word_pairs = nlp.read_csv(CSV_DIR + CSV_NAME)
    vocab = []
    for p in word_pairs:
        vocab.append(p[0])
        vocab.append(p[1])
    vocab = list(set(vocab))
    for w in vocab:
        sense_vectors[w] = []
    for k in word_vectors.keys():
        w = k.split("%")
        if ( w[0] in vocab ): #sense_vector_found
            sense_vectors[w[0]].append(word_vectors[k])
    test_sense_vectors()
