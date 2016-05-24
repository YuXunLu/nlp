#Optimization: C=1/2[ (w^* - \frac{1}{n} \sum_{j=1}^N s_j)^2 + \sum_{k=1}^N(s_j - s_j^*)^2]
import nlp_lib as nlp
import numpy as np
import scipy as sci

CSV_DIR = "../../csv/"
CSV_NAME = "word353sim.csv"
VECTOR_DIR = "../test_vector/"
VECTOR_NAME = "100_8.vec"
VECTOR_DIM = 100
MAX_ITER = 1
L_RATE = 0.001
alpha = 0.1
beta = 0.9
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
    print "Opt1 Global Approach", p_val

def cost_func(s):
    result = 0.0
    for l in s.lemmas():
        wd = l.name()
        dis = alpha * np.linalg.norm( word_pool[wd] - sense_vectors[s] )
        dis = np.power(dis,2)
        dis1 = beta * np.linalg.norm(sense_vectors[s] - sense_pool[s])
        dis1 = np.power(dis1,2)
    result = result + dis + dis1
    result = 0.5 * result
    return result
if __name__ == "__main__":
    print "VECTOR:", VECTOR_NAME
    word_vectors = nlp.read_word_vectors(VECTOR_DIR + VECTOR_NAME)
    word_pairs = nlp.read_csv(CSV_DIR + CSV_NAME)
    print "CSV", CSV_NAME
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
            sense_vectors[s] = sense_pool[s]
            for l in s.lemmas():
                word = str(l.name())
                word_hypernyms[word] = nlp.read_hypernyms(word)
                word_hyponyms[word] = nlp.read_hyponyms(word)
                word_synonyms[word] = nlp.read_synonyms(word)
                word_pool[word] = nlp.get_pooling(word, word_hypernyms, word_synonyms, word_hyponyms, word_vectors, VECTOR_DIM)

        word_pool[w] = nlp.get_pooling(w, word_hypernyms, word_synonyms, word_hyponyms, word_vectors, VECTOR_DIM)
    iter_num = 0
    sense_ind = []
    for w in vocab:
        for s in senses[w]:
            sense_ind.append(s)
    while (iter_num < MAX_ITER):
        for s in sense_ind:
            pre_cost = 9999999.0
            cost = cost_func(s)
            new_sense_vec = np.zeros(VECTOR_DIM)
            old_sense_vec = np.zeros(VECTOR_DIM)
            l_rate = L_RATE
            while ( cost < pre_cost):
                print "Sense",s ,"Iter",iter_num,"Cost",cost
                old_sense_vec = sense_vectors[s]
                term1 = np.zeros(VECTOR_DIM)
                term2 = np.zeros(VECTOR_DIM)
                for l in s.lemmas():
                    word = str(l.name())
                    #print "Lemma",word
                    #print "Vector",word_pool[word]
                    term1 = term1 - alpha * ( word_pool[word] - sense_vectors[s])
                    term2 = term2 + beta * (sense_vectors[s] - sense_pool[s])
#                print "term1",term1
#                print "term2",term2
                sense_vectors[s] = sense_vectors[s] - l_rate * (term1 + term2)
#                print "Sense_Vec", sense_vectors[s]
                pre_cost = cost
                cost = cost_func(s)
                if ( cost > pre_cost):
                    sense_vectors[s] = old_sense_vec
                    break;
                else:
                    l_rate = l_rate / (1.0 + 5e-6)
        iter_num = iter_num + 1
    test_sense_vectors()
