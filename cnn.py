from nltk.corpus import wordnet as wn
import nlp_lib as nlplib
import numpy as np
word_vectors = {}
new_word_vectors = {}
word_vector_dim = 100
word_pair_score = []
word_dictionary = []
word_hypernyms = {}
word_hyponyms = {}
word_synonyms = {}
#CNN part
weight_convolution = np.identity( 3, np.float64)
bias_vector = np.zeros( (1, word_vector_dim), np.float64)
learning_rate = 0.01
def df_dw(feature_map, vy, row_num):
    gradient = 0.0
    k = 0
    while ( k < word_vector_dim ):
        i = 0
        while ( i < len(feature_map) ):
            gradient = gradient + vy[k] * feature_map[row_num][k]
            i = i + 3
        k = k + 1
    n = len(feature_map) - 2.0
    gradient = gradient / n
    return gradient

def dg_dw(vx,feature_map,row_num):
    vx_len = np.dot(vx, np.transpose(vx) )
    vx_len = np.sqrt(vx_len)
    gradient = 0.0
    k = 0
    while ( k < word_vector_dim):
        i = 0
        while ( i < len(feature_map) ):
            gradient = gradient + feature_map[row_num][k]
            i = i + 3
        k = k + 1
    return gradient * vx_len

def compute_w_gradient(vx, vy, df_dw, dg_dw):
    vx_len = np.dot(vx, np.transpose(vx) )
    vy_len = np.dot(vy, np.transpose(vy) )
    vx_len = np.sqrt(vx_len)
    vy_len = np.sqrt(vy_len)

    f = np.dot(vx, np.transpose(vy) )
    g = vx_len

    factor1 = 1.0/(vx_len * vx_len * vy_len)
    gradient = (df_dw * g - f * dg_dw) * factor1
    return gradient
def update_parameters(bias, weight, feature_map, vx, vy):
    vx_len = np.dot(vx, np.transpose(vx) )
    vy_len = np.dot(vy, np.transpose(vy) )
    vx_len = np.sqrt(vx_len)
    vy_len = np.sqrt(vy_len)

    factor1 = 1.0/(vx_len * vx_len * vy_len)

    df_dw1 = df_dw(feature_map, vy, 0)
    df_dw2 = df_dw(feature_map, vy, 1)
    df_dw3 = df_dw(feature_map, vy, 2)

    dg_dw1 = dg_dw(vx, feature_map, 0)
    dg_dw2 = dg_dw(vx ,feature_map, 1)
    dg_dw3 = dg_dw(vx, feature_map, 2)

    #update gradient for convolution_weight's each row
    grad_w1 = compute_w_gradient(vx,vy,df_dw1,dg_dw1)
    grad_w2 = compute_w_gradient(vx,vy,df_dw2,dg_dw2)
    grad_w3 = compute_w_gradient(vx,vy,df_dw3,dg_dw3)
    
    i = 0
    while ( i < 3 ):
        weight[0,i] = weight[0,i] - learning_rate * grad_w1
        weight[1,i] = weight[1,i] - learning_rate * grad_w2
        weight[2,i] = weight[2,i] - learning_rate * grad_w3
        i = i + 1
    #update gradient for b, which is darned simpler.
    return bias, weight
def lost_function(vx,vy,score):
    dot_prod = np.dot(vx, np.transpose(vy) )
    vx_len = np.dot(vx, np.transpose(vx) )
    vy_len = np.dot(vy, np.transpose(vy) )
    vx_len = np.sqrt(vx_len)
    vy_len = np.sqrt(vy_len)
    bottom = vx_len*vy_len
    final_score = (dot_prod / bottom)
    lost_result = np.abs( final_score - score/(5.0) )
    return lost_result
def cnn_calc(w1,w2):
    feature_map = []
    hypernym_features = []
    hyponym_features = []
    synonym_features = []
    #STEP 1: BUILDING FEATURE MAP, width = 3
#    print "Build feature map"
    if (word_hypernyms.has_key(w1) ):
        for w in word_hypernyms[w1]:
            if (word_vectors.has_key(w) ):
                hypernym_features.append( word_vectors[w] )
    if (word_synonyms.has_key(w1) ):
        for w in word_synonyms[w1]:
            if (word_vectors.has_key(w)):
                synonym_features.append(word_vectors[w])
    if (word_hyponyms.has_key(w1) ):
        for w in word_hyponyms[w1]:
            if (word_vectors.has_key(w)):
                hyponym_features.append(word_vectors[w])
    i = 0
    while ( 0 < 1 ):
        if ( i < len(hypernym_features) ):
            feature_map.append(hypernym_features[i])
        else:
            break
        if ( i < len(synonym_features) ):
            feature_map.append(synonym_features[i])
        else:
            break
        if ( i < len(hyponym_features) ):
            feature_map.append(hyponym_features[i])
        else:
            break
        i = i + 1
    #STEP 2: CONVOLUTION STAGE
#    print "convolution"
    if ( len(feature_map) >= 3 ):
        #at least 3 nodes in feature map, which correspondent to a convolutional layer node
        conv_nodes_num = len(feature_map) - 2 # convolution nodes number
        convolution_result = []
        i = 0
        while ( i < conv_nodes_num ):
            node_feature = np.zeros( (3, word_vector_dim) , np.float64)
            node_feature[0] = feature_map[i]
            node_feature[1] = feature_map[i+1]
            node_feature[2] = feature_map[i+2]
            #convolution
            ADD_VEC = np.array( [1,1,1], np.float64 )
            result1 = np.dot(weight_convolution, node_feature)
            result2 = np.dot(ADD_VEC, result1)
            result3 = np.add(result2, bias_vector)
            this_node_result = result3
            convolution_result.append(this_node_result)
            i = i + 1
    else:
#        print "warning, convolution nodes insufficient","[word]:", w1
        return np.zeros( (1, word_vector_dim), np.float64 ), feature_map
    #STEP 3: POOLING
#    print "pooling"
    final_vector = np.zeros( (1, word_vector_dim), np.float64 )
    conv_nodes_N = np.array(conv_nodes_num, np.float64)
    for v in convolution_result:
        final_vector = np.add(final_vector, v)
    final_vector = np.divide(final_vector, conv_nodes_N)
    #return v_{x}^*, and feature map for weight updating.
    return final_vector, feature_map
def cnn_training():
    #initialize weight matrix and bias vector there!
    #weight_convolution = np.identity( 3, np.float64 )
    #bias_vector = np.zeros( (1, word_vector_dim), np.float64)
    global bias_vector
    global weight_convolution
    global learning_rate
    print "Training."
    sum_error = 0.0
    down_time = 0
    former_sum_error = 0.0
    while ( 1 > 0 ):
        sum_error = 0.0
        for word_pairs in word_pair_score:
            word_1 = word_pairs[0]
            word_2 = word_pairs[1]
            score = np.float32(float(word_pairs[2]))
            v_star, f_map = cnn_calc(word_1,word_2) #v^* and feature map
            if ( len(f_map) >= 3 ): #must have sufficient feature maps
                if (word_vectors.has_key(word_2) ):
#                    print "w1",word_1
#                    print "w2",word_2
#                    print "lost"
                    lost_single = lost_function(v_star, word_vectors[word_2], score)
                    sum_error = sum_error  + lost_single
#                    print "lost_single", lost_single
#                    print "accumulated_error", sum_error
#                    print "Updating parameters"
                    bias, weight = update_parameters(bias_vector, weight_convolution, f_map, v_star, word_vectors[word_2])
                    bias_vector = bias
                    weight_convolution = weight
        if (sum_error <= former_sum_error):
            down_time = down_time + 1
            learning_rate = learning_rate + 0.01
        else:
            down_time = 0
            learning_rate = learning_rate / 2.0
        if (down_time >= 5):
            break
        print "this time error", sum_error
        print "former time error", former_sum_error
        print "down_time", down_time
        print "weight mat", weight_convolution
        print "bias vec", bias_vector
        print "learning_rate", learning_rate
        former_sum_error = sum_error
if __name__=="__main__":
    print "read vector & score"
    word_vectors = nlplib.read_word_vectors("./test_vector/100_3.vec")
    word_pair_score = nlplib.read_csv("./csv/R&G-65.csv")
######Read hypernyms
    print "get hypernyms etc."
    for w_pair_score in word_pair_score:
        word_dictionary.append(w_pair_score[0])
        word_dictionary.append(w_pair_score[1])
#   remove dumplicated word, for we are searching hypernyms, hyponyms, synonyms according to the dictionary.
    word_dictionary = list(set(word_dictionary))
    for w in word_dictionary:
        word_hypernyms[w] = nlplib.read_hypernyms(w)
        word_hyponyms[w] = nlplib.read_hyponyms(w)
        word_synonyms[w] = nlplib.read_synonyms(w)
    print "start training"
    cnn_training()
