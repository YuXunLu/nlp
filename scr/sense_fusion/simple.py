import nlp_lib as nlp
import numpy as np
import scipy as sci
word_vectors = {}
word_pairs = []
target_file = "R&G-65.csv"
csv_path = "../../csv/"
vector_path = "../test_vector/"
vector_file = "80_10_punc.vec"
vector_dim = [80]
vector_win = [10]
def cos_function(v1,v2):
    result = 0.0
    tp = np.dot(v1, np.transpose(v2))
    btm = np.linalg.norm(v1) * np.linalg.norm(v2)
    result = tp/btm
    return result
if __name__ == "__main__":
    for dim in vector_dim:
        for win in vector_win:
            target_vector = str(dim) + "_" + str(win) + ".vec"
            word_vectors = nlp.read_word_vectors(vector_path + target_vector)
            print "current vec",target_vector
            print "current file", target_file
            word_pairs = nlp.read_csv(csv_path + target_file)
            m_score = []
            h_score = []
            for p in word_pairs:
                if ( word_vectors.has_key(p[0]) and word_vectors.has_key(p[1])):
                    h_score.append(float(p[2]))
                    m_score.append(cos_function(word_vectors[p[0]], word_vectors[p[1]]))
            p_val,p_dif = sci.stats.spearmanr(h_score,m_score)
            print "Spearmanr Single", str(p_val)

