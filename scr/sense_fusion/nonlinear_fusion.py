#Non-Linear Version of Sense Fusion
import nlp_lib as nlp
import numpy as np
import scipy as sci

CSV_DIR = "../../csv/"
CSV_NAME = "R&G-65.csv"
VECTOR_DIR = "../test_vector/"
VECTOR_NAME = "100_3.vec"
VECTOR_DIM = 100
L_RATE = 0.0005
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
        m_score = cos_function( word_pool[w1], word_pool[w2] )
        machine_score.append(m_score)
    p_val, p_rel = sci.stats.spearmanr(human_score, machine_score)
    print "Our old approach:", p_val
    machine_score = []
    for p in word_pairs:
        w1 = p[0]
        w2 = p[1]
        m_score = 0.0
        for s1 in senses[w1]:
            for s2 in senses[w2]:
                m_score = m_score + cos_function(sense_vectors[s1] , sense_vectors[s2])
        m_score = m_score / ( float( len(senses[w1] ) ) * float ( len(senses[w2])) )
        machine_score.append(m_score)

    p_val, p_rel = sci.stats.spearmanr(human_score, machine_score)
    print "My Approach", p_val

def cost_function(word_pool_vector, sense_vector):
    result = np.linalg.norm( word_pool_vector - sense_vector)
    result = result * result * 0.5
    return result
def calc_NN(word, p_w, p_u, p_v, p_b):
    result = np.zeros(VECTOR_DIM)
    hidden_result = []
#    print "para", para_w, para_u, para_v, para_b, word
    #from input to hidden layer node
    for s in senses[word]:
        hidden_res_s = np.tanh( np.dot(p_u[s], p_v[s]) + p_b[s] )
        hidden_result.append(hidden_res_s)
    tmp_s_vec = hidden_result
    i = 0
    while ( i < len(p_w) ):
        tmp_s_vec[i] = p_w[i] * hidden_result[i]
        i = i + 1
    hidden_result = np.dot(p_w, hidden_result)
    hidden_result = np.asarray(hidden_result) / float( len(senses[w]) )
    result = hidden_result
    return tmp_s_vec,result
def train_NN():
    for w in vocab:
        para_v = {}
        para_b = {}
        para_u = {}
        para_w = np.ones( len(senses[w]) )
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
#            para_v[s] = np.asarray(para_v[s])
            para_u[s] = np.random.randn(feature_num)
#            para_v[s] = np.asarray(para_v[s])
        #update parameters
        pre_vecs, s_star = calc_NN(w, p_w = para_w, p_u = para_u, p_v = para_v, p_b = para_b)
        cost = cost_function(word_pool[w], s_star)
        pre_cost = 10000
        first_cost = cost
        pre_s_star = s_star
        print "cost",cost
        while (  cost > (first_cost * 0.05)   ):
            grad_u = {}
            grad_b = {}
            s_vecs = []
            grad_w = np.zeros( len(senses[w]) )
            i = 0
            factor0 =  word_pool[w] - s_star
            for s in senses[w]:
                #compute gradu
                factor1 = para_w[i] * -1.0/ float(len(senses[w]))
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
                grad_u = grad_u * factor1
                grad_u = np.dot(factor0, grad_u)

                #compute gradb
                grad_b = np.zeros( (VECTOR_DIM, VECTOR_DIM) )
                j = 0
                while ( j < VECTOR_DIM ):
                    tan_part = np.tanh( np.dot ( para_u[s], tmp_v[:,j]) + para_b[s][j] )
                    tan_part = np.power(tan_part, 2)
                    grad_b[j,j] = 1.0 - tan_part
                    j = j + 1
                grad_b = grad_b * factor1
                grad_b = np.dot(factor0, grad_b)
                #update para_u, para_b and para_w
                para_b[s] = para_b[s] - L_RATE * grad_b
                para_u[s] = para_u[s] - L_RATE * grad_u
                para_w[i] = para_w[i] - L_RATE * np.dot(factor0, np.transpose(-pre_vecs[i] ) ) * -1.0 / len(senses[w])
                i = i + 1

            pre_cost = cost
            s_vecs, s_star = calc_NN(w, p_w = para_w, p_u = para_u, p_v = para_v, p_b = para_b)
            cost = cost_function(word_pool[w], s_star)
            print "pre_cost",pre_cost, "cost", cost
            if ( (pre_cost - cost) < 0.00000001 ):
                s_vecs = pre_vecs
                break
            pre_vecs = s_vecs
        i = 0
        for s in senses[w]:
            sense_vectors[s] = s_vecs[i]
            i = i + 1
    test_sense_vectors()
if __name__ == "__main__":
    word_vectors = nlp.read_word_vectors(VECTOR_DIR + VECTOR_NAME)
    print "LEARNING_RATE", L_RATE
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
            sense_hypernyms[s] = nlp.read_hypernyms_by_sense(s)
            sense_hyponyms[s] = nlp.read_hyponyms_by_sense(s)
            sense_synonyms[s] = nlp.read_synonyms_by_sense(s)
        word_pool[w] = nlp.get_pooling(w, word_hypernyms, word_synonyms, word_hyponyms, word_vectors)
    train_NN()
