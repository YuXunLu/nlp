import scipy
print 'Initialize enviroment for scipy, please wait'
scipy.test('fuck')
origin_vec = []
f_origin = open('RG.csv','r')
for l_origin in f_origin:
	origin_str = l_origin.split(',')
	origin_vec.append(float(origin_str[2]))
#print 'f_origin length is:'
#print len(origin_vec)
#raw_input('Press any key to continue')
max_sim = 0;
max_sim_file = '';
F_NAME1 = ['100']
F_NAME2 = ['3']
for name1 in F_NAME1:
	for name2 in F_NAME2:
		f_name = 'exp50.res'
		f_input = open(f_name,'r')
		input_vec = []
		for l_input in f_input:
			input_str = l_input.split(',')
#			print input_str
			del(input_str[0]) #del the word-pair
#			print float(input_str[0])
			input_vec.append(float(input_str[0]))
#		print 'input vec length is:',len(input_vec)
#		raw_input('Press anykey to continue')
		f_input.close()
		print f_name
		p = scipy.stats.pearsonr(origin_vec,input_vec)
		print p
		if p > max_sim:
			max_sim = p
			max_sim_file = f_name
print 'Max similarity is:',max_sim
print 'Correspodent file is:',max_sim_file		
