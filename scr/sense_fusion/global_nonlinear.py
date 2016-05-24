#Non-Linear Version of Sense Fusion
import nlp_lib as nlp
import numpy as np
import scipy as sci

CSV_DIR = "../../csv/"
CSV_NAME = "R&G-65.csv"
VECTOR_DIR = "../test_vector/"
VECTOR_NAME = "100_3.vec"
VECTOR_DIM = 100
L_RATE = 0.5
epsilon = 1e-3
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
        if ( (len(senses[w1]) > 0) and (len(senses[w2]) > 0)):
            m_score = m_score / ( float( len(senses[w1] ) ) * float ( len(senses[w2])) )
        if ( m_score > 0.0 ):
            machine_score.append(m_score)
            human_score.append(float(p[2]))

    p_val, p_rel = sci.stats.spearmanr(human_score, machine_score)
    print "NN Approach", p_val

def cost_function(word):
    result = 0.0
    for s in senses[word]:
        dis1 = np.linalg.norm(word_pool[word] - sense_vectors[s])
        dis1 = np.power(dis1, 2)
        dis = np.linalg.norm(sense_vectors[s] - sense_pool[s])
        dis = np.power(dis,2)
        result = result + dis + dis1
    result = result * 0.5
    return result
def calc_NN(word, p_w, p_u, p_v, p_b):
    result = np.zeros(VECTOR_DIM)
    hidden_result = {}
#    print "para", para_w, para_u, para_v, para_b, word
    #from input to hidden layer node
    for s in senses[word]:
        hidden_res_s = np.tanh( np.dot(p_u[s], p_v[s]) + p_b[s] )
        hidden_result[s] = hidden_res_s
    return hidden_result
def train_NN():
    for w in vocab:
        para_v = {}
        para_b = {}
        para_u = {}
        para_w = np.ones( len(senses[w]) )
        jump_this_word = 0
        #initialize parameters
        for s in senses[w]:
            para_b[s] = np.zeros(VECTOR_DIM)
            para_v[s] = []
            feature_num = 0
            for t in sense_hypernyms[s]:
                if ( word_vectors.has_key(t) ):
                    para_v[s].append( word_vectors[t])
            for t in sense_synonyms[s]:
                if ( word_vectors.has_key(t) ):
                    para_v[s].append( word_vectors[t] )
            for t in sense_hyponyms[s]:
                if ( word_vectors.has_key(t) ):
                    para_v[s].append( word_vectors[t] )
            feature_num = len(para_v[s])
            if ( feature_num < 1):
                print "Warning, sense",s,"word",w,"no features!"
                if (word_vectors.has_key(w) ):
                    para_v[s].append(word_vectors[w])
                else:
                    print "and its word also no vectors"
                    jump_this_word = 1
                    break
            para_u[s] = np.random.randn(feature_num)
        if (jump_this_word == 1):
            continue
        pre_vecs = calc_NN(w, p_w = para_w, p_u = para_u, p_v = para_v, p_b = para_b)
        for s in senses[w]:
            sense_vectors[s] = pre_vecs[s]
        cost = cost_function(w)
        pre_cost = 10000
        print "cost",cost
        ascent_t = 0
        while (  cost > 0   ):
            grad_u = {}
            grad_b = {}
            s_vecs = {}
            i = 0
            factor0 = np.zeros(VECTOR_DIM)
            factor1 = np.zeros(VECTOR_DIM)
            for s in senses[w]:
                factor0 = factor0 + (sense_vectors[s] - word_pool[w])
                factor1 = factor1 + (sense_vectors[s] - sense_pool[s])
            for s in senses[w]:
                #compute gradu
                grad_u = np.zeros( (VECTOR_DIM, len(para_u[s])) )
                tmp_v = np.asarray(para_v[s])
                j = 0
                k = 0
                while ( k < VECTOR_DIM ):
                    while ( j < len(para_u[s]) ):
                        tan_part = np.tanh( np.dot( para_u[s], tmp_v[:,k]) + para_b[s][k] )
                        tan_part = np.power(tan_part,2)
                        grad_u[k][j] = (1.0 - tan_part) * tmp_v[j,k]
                        j = j + 1
                    k = k + 1
                grad_u = np.dot(factor1 + factor0, grad_u)

                #compute gradb
                grad_b = np.zeros( (VECTOR_DIM, VECTOR_DIM) )
                j = 0
                while ( j < VECTOR_DIM ):
                    tan_part = np.tanh( np.dot ( para_u[s], tmp_v[:,j]) + para_b[s][j] )
                    tan_part = np.power(tan_part, 2)
                    grad_b[j,j] = 1.0 - tan_part
                    j = j + 1
                grad_b = np.dot(factor1 + factor0, grad_b)
                #update para_u, para_b and para_w
                para_b[s] = para_b[s] - L_RATE * grad_b
                para_u[s] = para_u[s] - L_RATE * grad_u
                i = i + 1
            pre_cost = cost
            s_vecs = calc_NN(w, p_w = para_w, p_u = para_u, p_v = para_v, p_b = para_b)
            for s in senses[w]:
                sense_vectors[s] = s_vecs[s]
            cost = cost_function(w)
            print "pre_cost",pre_cost, "cost", cost
            if ( (pre_cost - cost) <= epsilon ):
                for s in senses[w]:
                    sense_vectors[s] = pre_vecs[s]
                print "Word",w, "Cost",cost
                break
            pre_vecs = s_vecs
    test_sense_vectors()
if __name__ == "__main__":
    word_vectors = nlp.read_word_vectors(VECTOR_DIR + VECTOR_NAME)
    print "LEARNING_RATE", L_RATE
    print "VECTOR", VECTOR_NAME
    print "Dataset", CSV_NAME
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
            sense_hypernyms[s] = nlp.read_hypernyms_by_sense(s)
            sense_hyponyms[s] = nlp.read_hyponyms_by_sense(s)
            sense_synonyms[s] = nlp.read_synonyms_by_sense(s)
            sense_pool[s] = nlp.get_pooling(s, sense_hypernyms, sense_synonyms, sense_hyponyms, word_vectors, VECTOR_DIM)
            if ( word_vectors.has_key(w)):
                sense_pool[s] = sense_pool[s] + word_vectors[w]
            if (word_vectors.has_key(w)):
                sense_vectors[s] = word_vectors[w]
        word_pool[w] = nlp.get_pooling(w, word_hypernyms, word_synonyms, word_hyponyms, word_vectors, VECTOR_DIM)
    train_NN()
