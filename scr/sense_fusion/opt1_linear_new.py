#Optimization: C=1/2[ (w^* - \frac{1}{n} \sum_{j=1}^N s_j)^2 + \sum_{k=1}^N(s_j - s_j^*)^2]
import nlp_lib as nlp
import numpy as np
import scipy as sci

CSV_DIR = "../../csv/"
CSV_NAME = "R&G-65.csv"
VECTOR_DIR = "../test_vector/"
VECTOR_NAME = "100_3.vec"
VECTOR_DIM = 100
MAX_ITER = 20
L_RATE = 0.005
alpha = 1.0
beta = 1.0
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
        human_score.append( float(p[2]))
        m_score = cos_function(word_pool[w1], word_pool[w2])
        machine_score.append(m_score)
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
        m_score = m_score / ( float( len(senses[w1] ) ) * float ( len(senses[w2])) )
        if ( m_score > 0.0 ):
            machine_score.append(m_score)
            human_score.append(float(p[2]))
    p_val, p_rel = sci.stats.spearmanr(human_score, machine_score)
    print "Batch Opt1 Approach", p_val
def cost_func(word, sense_vecs):
    result = 0.0
    for s in senses[word]:
        dis = np.linalg.norm( word_pool[word] - sense_vecs[s] )
        dis = np.power(dis,2)
        dis1 = np.linalg.norm(sense_vecs[s] - sense_pool[s])
        dis1 = np.power(dis1,2)
        result = result + dis + dis1
    result = 0.5 * result
    return result
if __name__ == "__main__":
    print "VECTOR:", VECTOR_NAME
    word_vectors = nlp.read_word_vectors(VECTOR_DIR + VECTOR_NAME)
    word_pairs = nlp.read_csv(CSV_DIR + CSV_NAME)
    vocab = []
    for p in word_pairs:
        vocab.append(p[0].lower())
        vocab.append(p[1].lower())
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
            sense_pool[s] = nlp.get_pooling(s, sense_hypernyms,sense_synonyms,sense_hyponyms, word_vectors, VECTOR_DIM)
            if ( word_vectors.has_key(w)):
                sense_pool[s] = sense_pool[s] + word_vectors[w]
            sense_vectors[s] = word_vectors[w]
        word_pool[w] = nlp.get_pooling(w, word_hypernyms, word_synonyms, word_hyponyms, word_vectors, VECTOR_DIM)
    iter_num = 0
    while (iter_num < MAX_ITER):
        for w in vocab:
            pre_cost = 9999999.0
            cost = cost_func(w, sense_vectors)
            new_sense_vec = {}
            old_sense_vec = {}
            l_rate = L_RATE
            while ( cost < pre_cost ):
                print "Word",w,"Iter",iter_num,"Cost",cost
                for s in senses[w]:
                    old_sense_vec[s] = sense_vectors[s]
                for s in senses[w]:
                    term1 = np.zeros(VECTOR_DIM)
                    term2 = np.zeros(VECTOR_DIM)
                    for s1 in senses[w]:
                        term1 = term1 - ( word_pool[w] - sense_vectors[s1] )
                        term2 = term2 + ( sense_vectors[s1] - sense_pool[s1])
                    sense_vectors[s] = sense_vectors[s] - l_rate * (term1 + term2)
                pre_cost = cost
                cost = cost_func(w, sense_vectors)
                if ( cost > pre_cost):
                    for s1 in senses[w]:
                        sense_vectors[s1] = old_sense_vec[s1]
                    break;
                else:
                    l_rate = l_rate / (1.0+5e-6)
        iter_num = iter_num + 1
    test_sense_vectors()
