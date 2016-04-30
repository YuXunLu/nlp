#exp50: Hyponyms_synset_pooling + Hypernyms_sysnet_pooling + word_embedding
import public_func as pf
from nltk.corpus import wordnet as wn
word_list = []
word_score = []
word_vector = {}
word_hypernyms = {}
word_hyponyms = {}
word_fin_vector  = {}
word_synsets = {}
origin_vector_file_prefix = "./word2phrase_origin/vectors_origin_"
target_file = "R&G-65.csv"
vector_file_path = "./gloss_vectors/"
DIMENSIONS = ['100','200','300','400','500','600','700']
WINDOWS = ['3','4','5','6','7','8']
word_list = []
word_vector = {}
word_synsets = {}
word_hypernyms = {}
word_hyponyms = {}
word_fin_vector = {}
if __name__ == "__main__":
    #Read target word, word vectors, hypernyms, synsets, A.K.A Preprocessing
    pf.read_target_word_from_file(word_list, word_score, target_file)
    pf.read_word_hyponyms(word_list, word_hyponyms, pf.wn.NOUN)
    pf.read_word_hypernyms(word_list, word_hypernyms, pf.wn.NOUN)
    pf.read_synset_from_wordnet(word_list, word_synsets)
#Read synsets
    for w in word_list:
        word_synsets[w] = wn.synsets(w, pos = wn.NOUN)
        for hypernym in word_hypernyms[w]:
            word_synsets[hypernym] = wn.synsets(hypernym, pos = wn.NOUN)
        for hyponym in word_hyponyms[w]:
            word_synsets[hyponym] = wn.synsets(hyponym, pos = wn.NOUN)
    for dim in DIMENSIONS:
        for window in WINDOWS:
            word_vector = {}
            word_fin_vector = {}
            dest_vector_file = origin_vector_file_prefix + dim + "_" + window
            pf.read_wordvector(word_vector, dest_vector_file)
            output_name = target_file + "-" + dim + "_" + window + ".res"
####Processing Stage ####
#exp50: Hyponyms_synset_pooling + Hypernyms_sysnet_pooling + word_embedding
            print "Current Processing: Dimension",dim,"Window",window
            for w in word_list:
                hyponym_synset_pooling_vector = [0 for i in range(int(dim))]
                hypernym_synset_pooling_vector = [0 for i in range(int(dim))]
                hyponym_synset_pooling_vector = pf.get_synset_pooling(w, word_vector, word_hyponyms, dim, is_pooling = True)
                hypernym_synset_pooling_vector = pf.get_synset_pooling(w, word_vector, word_hypernyms, dim, is_pooling = True)
                word_fin_vector[w] = [0 for i in range(int(dim))]
                word_fin_vector[w] = pf.v_add(word_fin_vector[w], hypernym_synset_pooling_vector)
                word_fin_vector[w] = pf.v_add(word_fin_vector[w], hyponym_synset_pooling_vector)
                if ( word_vector.has_key(w) ):
                    word_fin_vector[w] = pf.v_add(word_fin_vector[w], word_vector[w] )
####Output the final result ######
            f_output = open(output_name,"w")
            i = 0
	    while i < len(word_list):
		word1 = word_list[i]
		word2 = word_list[i+1]
		if word_vector.has_key(word1):
			if word_vector.has_key(word2):
				sim = pf.cosine_sim(word_fin_vector[word1],word_fin_vector[word2])
                                if ( sim == -2.0 ):
                                    print "Warning, Something Error(divided by zero?)"
			else:
				print 'Warning',word2,'has no vector'
				sim = -2.0
		else:
			print 'Warning',word1,'has no vector'
			sim = -2.0		
		i = i+2
		f_output.write(word1 + '-' + word2 + ',' + str(sim) + '\n')
	    f_output.close()
