import math
from nltk.corpus import wordnet as wn
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
st = LancasterStemmer()
english_stopwords = stopwords.words('english')
english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
#dest_word: w in original file
#word_vector: word_vector
#pooling_vector: hypernym/hyponym array
#dim: Dimension
#is_pooling: if divided
def get_synset_pooling(dest_word, word_vector, pooling_vector, dim, is_pooling = True):
    return_vector = [0.0 for i in range(int(dim)) ]
    pooling_count = 0.0
    for word in pooling_vector[dest_word]:
        if ( word_vector.has_key(word) ):
            return_vector = v_add(return_vector, word_vector[word])
            pooling_count = pooling_count + 1.0
        else:
            print "[Warning],word",word,"does not found a vector of its related word",word
    if ( is_pooling == True ):
        if ( pooling_count > 0.0 ):
            return_vector = v_div(return_vector,pooling_count)
        return return_vector
    else:
        return return_vector
#gloss_vector: hypernym/hyponym array
def get_gloss_pooling(dest_word, word_vector, word_synsets, gloss_vector, dim, is_pooling = True):
    return_vector = [0.0 for i in range(int(dim)) ]
    pooling_count = 0.0
    for word in gloss_vector[dest_word]:
        for s in word_synsets[word]:
            filtered_gloss = get_filtered_gloss(s.definition() )
            for g in filtered_gloss:
                if(word_vector.has_key(g)):
                    return_vector = v_add(return_vector, word_vector[g])
                    pooling_count = pooling_count + 1.0
                else:
                    print "[Warning]word",word,"synset",s,"gloss",g,"has no vector"
    if( is_pooling == True):
        if ( pooling_count > 0.0 ):
            return_vector = v_div(return_vector, pooling_count)
        return return_vector
    else:
        return return_vector
def get_final_vector(gloss_matrix):
	result = []
	gloss_array = np.array(gloss_matrix)
	#Get the max vector of all row-vectors
	tmp_max = np.amax(gloss_array,0)
	#Get the min vector of all row-vectors
	tmp_min = np.amin(gloss_array,0)
	total_len = len(tmp_max)
	i = 0
	while i < total_len:
		if abs(tmp_max[i]) < abs(tmp_min[i]):
			result.append(tmp_min[i])
		else:
			result.append(tmp_max[i])
		i = i + 1
	return result
def get_filtered_gloss(raw_gloss):
	gloss1 = word_tokenize(raw_gloss) #tokenize
	gloss2 = [ word for word in gloss1 if not word in english_stopwords ] #remove stop words
	gloss3 = [ word for word in gloss2 if not word in english_punctuations ] #remove punctuations
	gloss4 = [ word for word in gloss3 if word.isalpha() ] #remove digits
	gloss5 = [ word.lower() for word in gloss4 ] #change it all to lowercase
	return gloss5
def get_final_vector(gloss_matrix):
	result = []
	gloss_array = np.array(gloss_matrix)
	#Get the max vector of all row-vectors
	tmp_max = np.amax(gloss_array,0)
	#Get the min vector of all row-vectors
	tmp_min = np.amin(gloss_array,0)
	total_len = len(tmp_max)
	i = 0
	while i < total_len:
		if abs(tmp_max[i]) < abs(tmp_min[i]):
			result.append(tmp_min[i])
		else:
			result.append(tmp_max[i])
		i = i + 1
	return result
def output_vector_single_word(fileName, targetWord, outputVector,dim):
	f_vec = open(fileName,"a")
	f_vec.write(targetWord + " ")
	i = 0
	for v in outputVector:
		f_vec.write(str(v))
		if ( i != dim ):
			f_vec.write(" ")
		i = i + 1
	f_vec.write("\n")
	f_vec.close()
def output_vector_with_label_and_gloss(fileName, outputVector,dim,wordList,wordHypernyms,wordHyponyms,wordGloss):
	f_vec = open(fileName,'w')
#Current Word#
	for w in wordList:
		i = 1
		if outputVector.has_key(w):
			f_vec.write(w + " ")
			for d in outputVector[w]:
				f_vec.write(str(d))
				if ( i != dim ):
					f_vec.write(' ')
				i = i + 1
			f_vec.write('\n')
		else:
			print w,"do not has a vector"
#Hypernyms
		print "output hypernym vectors"
		for hyper_w in wordHypernyms[w]:
			i = 1
			if outputVector.has_key(hyper_w):
				f_vec.write(hyper_w + " ")
				for d in outputVector[hyper_w]:
					f_vec.write(str(d))
					if ( i != dim ):
						f_vec.write(' ')
					i = i + 1
				f_vec.write('\n')
			else:
				print w,"do not has a vector"
		print "output hyponym vectors"
#Hyponyms
		for hypon_w in wordHyponyms[w]:
			i = 1
			if outputVector.has_key(hypon_w):
				f_vec.write(hypon_w + " ")
				for d in outputVector[hypon_w]:
					f_vec.write(str(d))
					if ( i != dim ):
						f_vec.write(' ')
					i = i + 1
				f_vec.write('\n')
			else:
				print w,"do not has a vector"
#Gloss
		print "output gloss vectors"
		for gloss_w in wordGloss[w]:
			i = 1
			if outputVector.has_key(gloss_w):
				f_vec.write(gloss_w + " ")
				for d in outputVector[gloss_w]:
					f_vec.write(str(d))
					if ( i != dim ):
						f_vec.write(' ')
					i = i + 1
				f_vec.write('\n')
			else:
				print w,"do not has a vector"
	f_vec.close()
def output_vector_with_label(fileName, outputVector,dim,wordList,wordHypernyms,wordHyponyms):
	f_vec = open(fileName,'w')
#Current Word#
	for w in wordList:
		i = 1
		if outputVector.has_key(w):
			f_vec.write(w + " ")
			for d in outputVector[w]:
				f_vec.write(str(d))
				if ( i != dim ):
					f_vec.write(' ')
				i = i + 1
			f_vec.write('\n')
		else:
			print w,"do not has a vector"
#Hypernyms
		print "output hypernym vectors"
		for hyper_w in wordHypernyms[w]:
			i = 1
			if outputVector.has_key(hyper_w):
				f_vec.write(hyper_w + " ")
				for d in outputVector[hyper_w]:
					f_vec.write(str(d))
					if ( i != dim ):
						f_vec.write(' ')
					i = i + 1
				f_vec.write('\n')
			else:
				print w,"do not has a vector"
		print "output hyponym vectors"
#Hyponyms
		for hypon_w in wordHyponyms[w]:
			i = 1
			if outputVector.has_key(hypon_w):
				f_vec.write(hypon_w + " ")
				for d in outputVector[hypon_w]:
					f_vec.write(str(d))
					if ( i != dim ):
						f_vec.write(' ')
					i = i + 1
				f_vec.write('\n')
			else:
				print w,"do not has a vector"
	f_vec.close()
def output_vector(fileName,outputVector,dim,wordList):
	f_vec = open(fileName,'w')
	for w in wordList:
		print 'Output Word Vector:',w
		i = 1
		if outputVector.has_key(w):
			for d in outputVector[w]:
				f_vec.write(str(d))
				if ( i != dim ):
					f_vec.write(' ')
				i = i + 1
			f_vec.write('\n')
	f_vec.close()
def output_label(fileName,wordList):
	f_label = open(fileName,'w')
	for w in wordList:
		f_label.write(w)
		f_label.write('\n')
	f_label.close()	
def read_target_word_from_file(wordList,wordScore, fileName):
	f_source = open(fileName,'r')
	for lines in f_source:
		target_words = lines.split(',')
		wordList.append(target_words[0].lower())
		wordList.append(target_words[1].lower())
		score = float(target_words[2])
		wordScore.append(score)
		wordScore.append(score)
	f_source.close()
def read_word_hypernyms(wordList,wordHypernyms, word_type = wn.NOUN):
	for w in wordList:
		word_synset = wn.synsets(w, pos = word_type)
		word_hypernym = []
		for s in word_synset:
			for h in s.hypernyms():
				for l in h.lemmas():
					word_hypernym.append(l.name())
		wordHypernyms[w] = word_hypernym
def read_word_hyponyms(wordList,wordHyponyms, word_type = wn.NOUN):
	for w in wordList:
		word_synset = wn.synsets(w, pos = word_type)
		word_hyponyms = []
		for s in word_synset:
			for h in s.hyponyms():
				for l in h.lemmas():
					word_hyponyms.append(l.name())
		wordHyponyms[w] = word_hyponyms
def read_target_word_from_RG(wordList, fileName = 'RG.csv'):
	f_RG = open(fileName,'r')
	for lines in f_RG:
		target_words = lines.split(',')
		del target_words[-1]
		set_words = target_words[0].split('-')
		wordList.append(set_words[0].lower())
		wordList.append(set_words[1].lower())
def read_synset_from_wordnet(wordList, wordSynsets, word_type = wn.NOUN):
	#This function would read all words, even it is duplicated in synsets, etc.
	for word in wordList:
		word_synset = wn.synsets(word, pos = word_type)
		synsets = []
		for s in word_synset:
			for l in s.lemmas():
				synsets.append(str(l.name()))
		wordSynsets[word] = synsets
#Function: Read word frequency
def read_word_frequency(word_frequency, f_name = 'word_list_new.txt'):
	f_wordf = open(f_name,'r')
	for lines in f_wordf:
		s_str = lines.split(' ')
		s_key = s_str[0]
		del s_str[0]
		word_frequency[s_key] = float(s_str[0])
	f_wordf.close()
#Function:Read word list and synsets
def read_wordlist(wordList,wordSynsets,f_name = 'synsets'):
	f_wordlist = open(f_name,'r')
	for lines in f_wordlist:
		s_str = lines.split(' ')
		s_key = s_str[0]
		del s_str[0]
		#HERE we need to decied whether we should remove the target word
		#This line move duplicated key word
		s_synset = [words.lower() for words in s_str if words != s_key]
		s_synset.append(s_key)
		#add elements in correspondent dictionary
		wordSynsets[s_key] = s_synset
		wordList.append(s_key)
	f_wordlist.close()

#Function: Read correspondent vector for each word
def read_wordvector(wordVector,f_name):
	f_vector = open(f_name,'r')
	for lines in f_vector:
		s_str = lines.split(' ')
		s_key = s_str[0]
		del s_str[0] #the key
		del s_str[-1]
		wordVector[s_key] = map(float,s_str)
	f_vector.close()
def v_multiple_scalar(v,scalar):
	result = [0.0 for i in range(len(v))]
	i = 0
	while i < len(v):
		result[i] = scalar * v[i]
		i = i + 1
	return result
def v_add(u,v):
	result = u
	i = 0
	while i < len(v):
		result[i] = result[i] + v[i]
		i = i + 1
	return result
def v_lamda_add(u,v,lamda):
	result = u
	i = 0
	while i < len(result):
		result[i] = result[i] * (1.0 - lamda)
		i = i + 1
	i = 0
	while i < len(v):
		result[i] = result[i] + lamda * v[i]
		i = i + 1
	return result
def v_div(u,scalar):
	result = u
	i = 0
	while i < len(result):
		result[i] = result[i] / scalar
		i = i + 1
	return result
def get_synset_sum_vector(keyword, wordVector, wordSynset):
	result = [0.0 for i in range(len(wordVector[keyword]))]
	for s in wordSynset[keyword]:
		s_count = 1.0
		if wordVector.has_key(s):
			s_count = s_count + 1.0
			result = v_add(result,wordVector[s])
		else:
			print 'Warning,word:',keyword,'do not has vector',s
	return result
#The pooling
def get_synset_vector(keyword, wordVector, wordSynset):
	result = [0.0 for i in range(len(wordVector[keyword]))]
	for s in wordSynset[keyword]:
		s_count = 1.0
		if wordVector.has_key(s):
			s_count = s_count + 1.0
			result = v_add(result,wordVector[s])
		else:
			print 'Warning,word:',keyword,'do not has vector',s
	result = v_div(result,s_count)
	return result
def cosine_sim(u,v):
	up = 0.0
	down = 0.0
	result = 0.0
	u_down = 0.0
	v_down = 0.0	
	i = 0
	while i < len(u):
		up = up + u[i] * v[i]
		i = i + 1
	i = 0
	while i < len(u):
		u_down = u_down + u[i] * u[i]
		v_down = v_down + v[i] * v[i]
		i = i + 1
	down = math.sqrt(u_down) * math.sqrt(v_down)
	if ( down == 0.0 ):
		result = -2.0
	else:
		result = up/down
	return result
