#Simple-Linear Version of Sense Fusion
import nlp_lib as nlp
import numpy as np
import scipy as sci

CSV_DIR = "../../csv/"
CSV_NAME = "WordSim353.csv"
VECTOR_DIR = "../test_vector/"
VECTOR_NAME = "100_6.vec"
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
sense_pool = {}
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
        w1 = p[0].lower()
        w2 = p[1].lower()
        if (word_vectors.has_key(w1) and word_vectors.has_key(w2) ):
            human_score.append( float(p[2]) )
            m_score = cos_function( word_vectors[w1], word_vectors[w2] )
            machine_score.append(m_score)
    p_val, p_rel = sci.stats.spearmanr(human_score, machine_score)
    print "Sense-Agnostic approach:", p_val
    human_score = []
    machine_score = []
    for p in word_pairs:
        w1 = p[0].lower()
        w2 = p[1].lower()
        if (word_pool.has_key(w1) and word_pool.has_key(w2) ):
            m_score = cos_function(word_pool[w1], word_pool[w2])
            if (m_score > 0.0):
                machine_score.append(m_score)
                human_score.append(float(p[2]))
    p_val, p_rel = sci.stats.spearmanr(human_score, machine_score)
    print "Sense-Agnostic Pooling Approach:", p_val
    machine_score = []
    human_score = []
    for p in word_pairs:
        w1 = p[0].lower()
        w2 = p[1].lower()
        m_score = 0.0
        for s1 in senses[w1]:
            for s2 in senses[w2]:
                if (sense_vectors.has_key(s1) and sense_vectors.has_key(s2)):
                    m_score = m_score + cos_function(sense_vectors[s1] , sense_vectors[s2])
        if ( len(senses[w1]) > 0 and len(senses[w2]) > 0):
            m_score = m_score / ( float( len(senses[w1] ) ) * float ( len(senses[w2])) )
        if ( m_score > 0.0):
            machine_score.append(m_score)
            human_score.append(float(p[2]))
    p_val, p_rel = sci.stats.spearmanr(human_score, machine_score)
    print "Opt2 Approach", p_val

if __name__ == "__main__":
    print VECTOR_NAME
    word_vectors = nlp.read_word_vectors(VECTOR_DIR + VECTOR_NAME)
    word_pairs = nlp.read_csv(CSV_DIR + CSV_NAME)
    vocab = []
    for p in word_pairs:
        vocab.append(p[0].lower())
        vocab.append(p[1].lower())
    vocab = list(set(vocab))
    new_sense_vecs = {}
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
            sense_pool[s] = nlp.get_pooling(s, sense_hypernyms, sense_synonyms, sense_hyponyms, word_vectors, VECTOR_DIM)
            if ( word_vectors.has_key(w)):
                sense_vectors[s] = sense_vectors[s] + word_vectors[w]
                sense_pool[s] = sense_pool[s] + word_vectors[w]
        word_pool[w] = nlp.get_pooling(w, word_hypernyms, word_synonyms, word_hyponyms, word_vectors, VECTOR_DIM)
        if ( len(senses[w]) == 1 ):
            for s in senses[w]:
                new_sense_vecs[s] = 0.5 * (sense_pool[s] + word_pool[w])
        else:
            for s in senses[w]:
                s_vec = np.zeros(VECTOR_DIM)
                s_div = len(senses[w])
                term1 = np.zeros(VECTOR_DIM)
                term2 = np.zeros(VECTOR_DIM)
                for s1 in senses[w]:
                    if (s == s1):
                        continue
                    term1 = term1 + ( -1.0/s_div ) * sense_vectors[s1]
                    term2 = term2 + sense_vectors[s1] - sense_pool[s1]
                s_vec = (word_pool[w] - term1 + term2) /  (s_div / (1.0-s_div))
                new_sense_vecs[s] = s_vec
    for w in vocab:
        for s in senses[w]:
            if (new_sense_vecs.has_key(s)):
                sense_vectors[s] = new_sense_vecs[s]
            else:
                print "syn",s,"not in new_sense"
                if ( sense_vectors.has_key(s) ):
                    print "but sense vectors do have key"
                print "len",len(senses[w])
    test_sense_vectors()
