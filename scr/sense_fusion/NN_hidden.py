#Hidden Layer version: Non-Linear Version of Sense Fusion
import nlp_lib as nlp
import numpy as np
import scipy as sci

CSV_DIR = "../../csv/"
CSV_NAME = "R&G-65.csv"
VECTOR_DIR = "../test_vector/"
VECTOR_NAME = "100_3.vec"
VECTOR_DIM = 100
L_RATE = 0.5
epsilon = 1e-6
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
    print "NN Hidden Approach", p_val

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
def calc_NN(word, p_u1, p_u2, p_u3, p_v1, p_v2, p_v3, p_b1, p_b2, p_b3):
    result = {}
    hidden_layer = {}
#    print "para", para_w, para_u, para_v, para_b, word
    #from input to hidden layer node
    for s in senses[word]:
        hide_node1 = np.dot(p_u1[s], p_v1[s]) + p_b1[s]
        hide_node2 = np.dot(p_u2[s], p_v2[s]) + p_b2[s]
        hide_node3 = np.dot(p_u3[s], p_v3[s]) + p_b3[s]
        hide_node1 = np.tanh(hide_node1)
        hide_node2 = np.tanh(hide_node2)
        hide_node3 = np.tanh(hide_node3)
        result[s] = (1.0/3.0) * (hide_node1 + hide_node2 + hide_node3)
    return result
def train_NN():
    for w in vocab:
        para_v1 = {}  #hypernym feature mat
        para_v2 = {}  #synonym feature mat
        para_v3 = {}  #hyponym feature mat

        para_b1 = {}  #hypernym bias vec
        para_b2 = {}  #synonym bias vec
        para_b3 = {}  #hyponym bias vec

        para_u1 = {} #hypernym weight
        para_u2 = {} #synonym weight
        para_u3 = {} #hyponym weight

        para_w = np.ones( len(senses[w]) ) 
        jump_this_word = 0
        #initialize parameters
        for s in senses[w]:
            para_b1[s] = np.zeros(VECTOR_DIM)
            para_b2[s] = np.zeros(VECTOR_DIM)
            para_b3[s] = np.zeros(VECTOR_DIM)

            para_v1[s] = []
            para_v2[s] = []
            para_v3[s] = []
            feature_num = 0
            for t in sense_hypernyms[s]:
                if ( word_vectors.has_key(t) ):
                    para_v1[s].append( word_vectors[t])
            for t in sense_synonyms[s]:
                if ( word_vectors.has_key(t) ):
                    para_v2[s].append( word_vectors[t] )
            for t in sense_hyponyms[s]:
                if ( word_vectors.has_key(t) ):
                    para_v3[s].append( word_vectors[t] )
            hyper_num = len(para_v1[s])
            syn_num = len(para_v2[s])
            hypon_num = len(para_v3[s])
            if ( (hyper_num < 1) or (syn_num < 1) or (hypon_num < 1)):
                if (word_vectors.has_key(w) ):
                    if (hyper_num < 1 ):
                        print "Warning, word",w,"no hypernyms, using word embedding instead"
                        para_v1[s].append(word_vectors[w])
                    if (syn_num < 1 ):
                        print "Warning, word",w,"no synonym, using word embedding instead"
                        para_v2[s].append(word_vectors[w])
                    if (hypon_num < 1):
                        print "Warning, word",w,"no hyponym, using word embedding instead"
                        para_v3[s].append(word_vectors[w])
                else:
                    print "None of word",w,"'s related word in vector file,:("
                    jump_this_word = 1
                    break
            hyper_num = len(para_v1[s])
            syn_num = len(para_v2[s])
            hypon_num = len(para_v3[s])
            para_u1[s] = np.random.randn(hyper_num)
            para_u2[s] = np.random.randn(syn_num)
            para_u3[s] = np.random.randn(hypon_num)
        if (jump_this_word == 1):
            continue
        pre_vecs = calc_NN(w, p_u1 = para_u1, p_u2 = para_u2, p_u3 = para_u3, p_v1 = para_v1, p_v2 = para_v2, p_v3 = para_v3, p_b1 = para_b1, p_b2 = para_b2, p_b3 = para_b3)
        for s in senses[w]:
            sense_vectors[s] = pre_vecs[s]
        cost = cost_function(w)
        pre_cost = 10000
        print "cost",cost
        ascent_t = 0
        while (  cost > 0   ):
            s_vecs = {}
            i = 0
            factor0 = np.zeros(VECTOR_DIM)
            factor1 = np.zeros(VECTOR_DIM)
            for s in senses[w]:
                factor0 = factor0 + (sense_vectors[s] - word_pool[w])
                factor1 = factor1 + (sense_vectors[s] - sense_pool[s])
            for s in senses[w]:
                #compute gradu
                grad_u1 = np.zeros( (VECTOR_DIM, len(para_u1[s])) )
                grad_u2 = np.zeros( (VECTOR_DIM, len(para_u2[s])) )
                grad_u3 = np.zeros( (VECTOR_DIM, len(para_u3[s])) )

                tmp_v1 = np.asarray(para_v1[s])
                tmp_v2 = np.asarray(para_v2[s])
                tmp_v3 = np.asarray(para_v3[s])

                j = 0
                k = 0
                while ( k < VECTOR_DIM ):
                    while ( j < len(para_u1[s]) ):
                        u_1_tan_part = np.tanh( np.dot( para_u1[s], tmp_v1[:,k]) + para_b1[s][k] )
                        u_1_tan_part = np.power(u_1_tan_part,2)
                        grad_u1[k][j] = (1.0 - u_1_tan_part) * tmp_v1[j,k]
                        j = j + 1
                    k = k + 1
                grad_u1 = np.dot(factor1 + factor0, grad_u1)
                i = 0
                k = 0
                while ( k < VECTOR_DIM ):
                    while ( j < len(para_u2[s]) ):
                        u_2_tan_part = np.tanh( np.dot( para_u2[s], tmp_v2[:,k]) + para_b2[s][k])
                        u_2_tan_part = np.power( u_2_tan_part, 2)
                        grad_u2[k][j] = (1.0 - u_2_tan_part) * tmp_v2[j,k]
                        j = j + 1
                    k = k + 1
                grad_u2 = np.dot(factor1 + factor0, grad_u2)

                i = 0
                k = 0
                while ( k < VECTOR_DIM ):
                    while ( j < len(para_u3[s]) ):
                        u_3_tan_part = np.tanh( np.dot( para_u3[s], tmp_v3[:,k]) + para_b3[s][k])
                        u_3_tan_part = np.power( u_3_tan_part, 2)
                        grad_u3[k][j] = (1.0 - u_3_tan_part) * tmp_v3[j,k]
                        j = j + 1
                    k = k + 1
                grad_u3 = np.dot(factor1 + factor0, grad_u3)

                #compute gradb
                grad_b1 = np.zeros( (VECTOR_DIM, VECTOR_DIM) )
                j = 0
                while ( j < VECTOR_DIM ):
                    b1_tan_part = np.tanh( np.dot ( para_u1[s], tmp_v1[:,j]) + para_b1[s][j] )
                    b1_tan_part = np.power(b1_tan_part, 2)
                    grad_b1[j,j] = 1.0 - b1_tan_part
                    j = j + 1
                grad_b1 = np.dot(factor1 + factor0, grad_b1)

                grad_b2 = np.zeros( (VECTOR_DIM, VECTOR_DIM) )
                j = 0
                while ( j < VECTOR_DIM ):
                    b2_tan_part = np.tanh( np.dot( para_u2[s], tmp_v2[:,j]) + para_b2[s][j] )
                    b2_tan_part = np.power(b2_tan_part, 2)
                    grad_b2[j,j] = 1.0 - b2_tan_part
                    j = j + 1
                grad_b2 = np.dot(factor1 + factor0, grad_b2)

                grad_b3 = np.zeros( (VECTOR_DIM, VECTOR_DIM) )
                j = 0
                while ( j < VECTOR_DIM ):
                    b3_tan_part = np.tanh( np.dot( para_u3[s], tmp_v3[:,j]) + para_b3[s][j])
                    b3_tan_part = np.power(b3_tan_part,2)
                    grad_b3[j][j] = 1.0 - b3_tan_part
                    j = j + 1
                grad_b3 = np.dot(factor1 + factor0, grad_b3)
                #update para_u, para_b and para_w
                para_b1[s] = para_b1[s] - L_RATE * 1.0/3.0 * grad_b1
                para_b2[s] = para_b2[s] - L_RATE * 1.0/3.0 * grad_b2
                para_b3[s] = para_b3[s] - L_RATE * 1.0/3.0 * grad_b3
                para_u1[s] = para_u1[s] - L_RATE * 1.0/3.0 * grad_u1
                para_u2[s] = para_u2[s] - L_RATE * 1.0/3.0 * grad_u2
                para_u3[s] = para_u3[s] - L_RATE * 1.0/3.0 * grad_u3
                i = i + 1
            pre_cost = cost
            s_vecs = calc_NN(w, p_u1 = para_u1, p_u2 = para_u2, p_u3 = para_u3,  p_v1 = para_v1, p_v2 = para_v2, p_v3 = para_v3, p_b1 = para_b1, p_b2 = para_b2, p_b3 = para_b3)
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
