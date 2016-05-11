from nltk.corpus import wordnet as wn
import nlp_lib as nlp_lib
import numpy as np
import scipy as sci
VECTOR_DIR ="../test_vector/"
VECTOR_NAME = "100_3.vec"
CSV_DIR = "../../csv/"
CSV_NAME = "M&C-30.csv"
VECTOR_DIM = 100
LEARNING_RATE = 0.005

vocab = []
word_pairs = []
word_senses = {}
word_sense_hyponyms = {}
word_sense_hypernyms = {}
word_sense_synonyms = {}
word_sense_vectors = {}
word_sense_full_pooling = {}

word_hyponyms = {}
word_hypernyms = {}
word_vectors = {}
word_synonyms = {}

word_final_vectors = {}
word_hypon_pooling = {}
word_sense_hypon_pooling = {}
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
def get_full_pooling_sense(s):
    result = np.zeros(VECTOR_DIM)
    i = 0.0
    if (len(word_sense_hypernyms[s]) > 0):
        for hyper in word_sense_hypernyms[s]:
            if word_vectors.has_key(hyper):
                result = result + word_vectors[hyper]
                i = i + 1.0
    if (len(word_sense_hyponyms[s]) > 0 ):
        for hypon in word_sense_hyponyms[s]:
            if word_vectors.has_key(hypon):
                result = result + word_vectors[hypon] 
                i = i + 1.0
    if (len(word_sense_synonyms[s] ) > 0 ):
        for syn in word_sense_synonyms[s]:
            if word_vectors.has_key(syn):
                result = result + word_vectors[syn]
                i = i + 1.0
    if ( ( i - 0.0 ) > 0.0 ):
        result = result / i
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

#calculating the sense vector
def CNN_calc(sense, word, feature_map, weight_mat):
    #the convolutional node # = feature # - 2
    conv_num = len(feature_map) - 2
    node_feature = np.zeros( (3, VECTOR_DIM) )
    conv_result = []
    result = np.zeros(VECTOR_DIM)
    i = 0
    #convolution step
    while ( i < conv_num ):

        res = np.zeros(VECTOR_DIM)
#        print "nodef",node_feature
#        print "fmap",feature_map[i]
        node_feature[0] = feature_map[i]
        node_feature[1] = feature_map[i+1]
        node_feature[2] = feature_map[i+2]     

        t_res = np.dot(weight_mat, node_feature)
        res = t_res[0] + t_res[1] + t_res[2]

        conv_result.append(res)
        i = i + 1
    #pooling step
    i = 0
    while ( i < len(conv_result) ):
        result = result + conv_result[i]
        i = i + 1
    result = result / len(conv_result)
    return result

def margin_function(word, sense, sense_vector):
    result = -1.0
    if ( (word_final_vectors.has_key(word)) and (word_vectors.has_key(word)) ):
        left = np.linalg.norm( word_final_vectors[word] - word_vectors[word] )
        left = left * left
        right = np.linalg.norm( word_sense_full_pooling[sense]  - sense_vector)
        right = right * right
        result = right - left
        print "left: v* to word_hypon_pooling:",left
        print "right: v_s* to sense_vector",right
    return result

#using cnn train sense vector
def train_CNN(sense, word):
    sense_vector = np.zeros(VECTOR_DIM)
    paddle = np.random.rand(VECTOR_DIM)
    weight_mat = np.random.randn(3,3)
    feature_map = []

    feature_map.append(paddle)

    #step1: build feature map - to full pooling!
    vec_hypon = []
    vec_hyper = []
    vec_syn = []

    i = 0
    for hyper in word_sense_hypernyms[sense]:
        if (word_vectors.has_key(hyper) ):
            vec_hyper.append( word_vectors[hyper] )
    for syn in word_sense_synonyms[sense]:
        if (word_vectors.has_key(syn) ):
            vec_syn.append( word_vectors[syn] )
    for hypon in word_sense_hyponyms[sense]:
        if (word_vectors.has_key(hypon) ):
            vec_hypon.append( word_vectors[hypon] )
    
    i = 0
    while (  ( i < len(vec_hyper) ) and ( i < len(vec_syn) ) and ( i < len(vec_hypon) ) ):
        feature_map.append(vec_hyper[i] )
        feature_map.append(vec_syn[i] )
        feature_map.append(vec_hypon[i] )
        i = i + 1

    feature_map.append(paddle)

    num_feature_map = len(feature_map)

    #step 2: calc cnn
    if ( num_feature_map > 2 ):
        tmp_sense_vector = CNN_calc(sense, word, feature_map, weight_mat)
#        print "word",word,"sense",sense
        m_value = margin_function(word, sense, tmp_sense_vector)
        pre_m_value = m_value
#        print "marigin_func", m_value
#        print "w_mat"
#        print weight_mat
        stop_flag = 0
        #gradient descent
        while( ( m_value > 0.0 ) and (stop_flag == 0 ) ):
#            print "word",word,"sense",sense
#            print "w_mat"
#            print weight_mat
            #update each weight
            v_factor1 = []
            i = 0
            while ( i < VECTOR_DIM ):
                v_factor1.append( word_final_vectors[word][i] - tmp_sense_vector[i] )
                i = i + 1
            
            v_factor1 = np.asarray(v_factor1)
            conv_num = num_feature_map - 2.0
            
            i = 0
            #update for dyt_dwj1, aka for 1st column in weight_mat
            dy_dwj1 = []
            while ( i < VECTOR_DIM ):
                j = 0
                tmp_grad = 0.0
                while ( j < num_feature_map ):
                    tmp_grad = tmp_grad + feature_map[j][i]
                    j = j + 3
                tmp_grad = -tmp_grad / conv_num
                dy_dwj1.append(tmp_grad)
                i = i + 1
            dy_dwj1 = np.asarray(dy_dwj1)
            grad_dwj1 = np.dot(dy_dwj1, np.transpose(v_factor1) )
            
            i = 0
            while ( i < 3 ):
                weight_mat[i][0] = weight_mat[i][0] - LEARNING_RATE * grad_dwj1
                i = i + 1

            #update for dyt_dwj2, aka for 2nd column in weight_mat
            i = 0
            dy_dwj2 = []
            while ( i < VECTOR_DIM ):
                j = 1
                tmp_grad = 0.0
                while ( j < num_feature_map ):
                    tmp_grad = tmp_grad + feature_map[j][i]
                    j = j + 3
                tmp_grad = -tmp_grad / conv_num
                dy_dwj2.append(tmp_grad)
                i = i + 1
            dy_dwj2 = np.asarray(dy_dwj2)
            grad_dwj2 = np.dot(dy_dwj2, np.transpose(v_factor1) )
            i = 0
            while ( i < 3 ):
                weight_mat[i][1] = weight_mat[i][1] - LEARNING_RATE * grad_dwj2
                i = i + 1

            #update for dyt_dwj3, aka for 3rd column in weight_mat
            i = 0
            dy_dwj3 = []
            while ( i < VECTOR_DIM ):
                j = 2
                tmp_grad = 0.0
                while ( j < num_feature_map):
                    tmp_grad = tmp_grad + feature_map[j][i]
                    j = j + 3
                tmp_grad = -tmp_grad/ conv_num
                dy_dwj3.append(tmp_grad)
                i = i + 1
            dy_dwj3 = np.asarray(dy_dwj3)
            grad_dwj3 = np.dot(dy_dwj3, np.transpose(v_factor1) )
            i = 0
            while ( i < 3 ):
                weight_mat[i][2] = weight_mat[i][2] - LEARNING_RATE * grad_dwj3
                i = i + 1


            #re-calculate
            pre_sense_vector = tmp_sense_vector
            tmp_sense_vector = CNN_calc(sense, word, feature_map, weight_mat)
            m_value = margin_function(word, sense, tmp_sense_vector)
            if ( m_value > pre_m_value ):
                stop_flag = 1
                tmp_sense_vector = pre_sense_vector
            pre_m_value = m_value
#            print "marigin_function", m_value
        sense_vector = tmp_sense_vector
    return sense_vector
#build and train sense vectors
def training_sense_vectors():
    for w in vocab:
        if ( len(word_senses[w]) > 0 ): #polynonmy
            for s in word_senses[w]:
                sense_vector = train_CNN(s, w)
                word_sense_vectors[s] = sense_vector
def test_sense_vectors():
    human_score = []
    machine_score = []

    #Our old approach
    for w_pair in word_pairs:
        w1 = w_pair[0].lower()
        w2 = w_pair[1].lower()
        h_score = float(w_pair[2])

        v1 = word_final_vectors[w1]
        v2 = word_final_vectors[w2]
        m_score = np.dot(v1, np.transpose(v2) ) / ( np.linalg.norm(v1) * np.linalg.norm(v2) )
        
        human_score.append(h_score)
        machine_score.append(m_score)

    sp1_val, sp1_rel = sci.stats.spearmanr(human_score, machine_score)

    print "Our pooling approach Spearman rel", sp1_val

    human_score = []
    machine_score = []

    #My Approach
    for w_pair in word_pairs:
        w1 = w_pair[0].lower()
        w2 = w_pair[1].lower()
        w1_s_vec = []
        w2_s_vec = []
        num_1 = 0
        m_score = 0.0
        for s in word_senses[w1]:
            if (word_sense_vectors.has_key(s) ):
                if ( np.linalg.norm(word_sense_vectors[s]) > 0 ): #not a zero vector
                    w1_s_vec.append( word_sense_vectors[s] )
        for s in word_senses[w2]:
            if (word_sense_vectors.has_key(s) ):
                if ( np.linalg.norm(word_sense_vectors[s]) > 0 ):#not a zero vec
                    w2_s_vec.append( word_sense_vectors[s] )
        if ( (len(w1_s_vec) > 0 ) and ( len(w2_s_vec) > 0 ) ):
            for v1 in w1_s_vec:
                for v2 in w2_s_vec:
                    m_score = m_score + np.dot(v1, np.transpose(v2) ) / ( np.linalg.norm(v1) * np.linalg.norm(v2) )

            m_score = m_score / ( len(w1_s_vec) * len(w2_s_vec) )
            machine_score.append(m_score)
        #w1 multiple meanings while w2 not
        if ( (len(w1_s_vec) > 0 ) and ( len (w2_s_vec) == 0 ) ):
            for v1 in w1_s_vec:
                m_score = m_score + np.dot(v1, np.transpose(word_vectors[w2]) ) / (np.linalg.norm(v1) * np.linalg.norm(word_vectors[w2]))

            m_score = m_score / len(w1_s_vec)
            machine_score.append(m_score)

        if ( (len(w2_s_vec) > 0 ) and ( len (w1_s_vec) == 0 ) ):
            for v2 in w2_s_vec:
                m_score = m_score + np.dot(v2, np.transpose(word_vectors[w1]) ) / (np.linalg.norm(v2) * np.linalg.norm(word_vectors[w1]))

            m_score = m_score / len(w2_s_vec)
            machine_score.append(m_score)

        if ( (len(w2_s_vec) == 0 ) and ( len (w1_s_vec) == 0 ) ):
            m_score = np.dot( word_vectors[w1], word_vectors[w2] ) / ( np.linalg.norm(word_vectors[w1]) * np.linalg.norm(word_vectors[w2]))
            machine_score.append(m_score)

        
        h_s = float ( w_pair[2] )
        human_score.append(h_s)
    #calc pearson
    p_rel, p_val = sci.stats.spearmanr(human_score, machine_score)
    print "My Approach Spearman Correlation Factor:",p_rel

if __name__ == "__main__":
    word_pairs = nlp_lib.read_csv( CSV_DIR + CSV_NAME )
    for w_pair in word_pairs:
        vocab.append( w_pair[0].lower() )
        vocab.append( w_pair[1].lower() )
    #remove duplicated words in csv file
    vocab = list(set(vocab))
    #read word senses
    for w in vocab:
        word_senses[w] = nlp_lib.read_senses(w)
    #read word senses' hyponyms
    for w in vocab:
        for s in word_senses[w]:
            word_sense_hyponyms[s] = nlp_lib.read_hyponyms_by_sense(s)
            word_sense_hypernyms[s] = nlp_lib.read_hypernyms_by_sense(s)
            word_sense_synonyms[s] = nlp_lib.read_synonyms_by_sense(s)
            word_sense_full_pooling[s] = get_full_pooling_sense(s)
    #read for retrofitting
    for w in vocab:
        word_hyponyms[w] = nlp_lib.read_hyponyms(w)
        word_hypernyms[w] = nlp_lib.read_hypernyms(w)
        word_synonyms[w] = nlp_lib.read_synonyms(w)
    #read word vectors

    word_vectors = nlp_lib.read_word_vectors(VECTOR_DIR + VECTOR_NAME)

    for w in vocab:
        word_final_vectors[w] = get_pooling(w)
    training_sense_vectors()
    #calculate pearson similarity
    test_sense_vectors()
