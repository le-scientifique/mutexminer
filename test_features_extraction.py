import csv
import pickle
import operator
train_dict = {}
def sort_dict(x):
	# x = {1: 2, 3: 4, 4: 3, 2: 1, 0: 0}
	sorted_x = sorted(x.items(), key=operator.itemgetter(1),reverse=True)
	return sorted_x

chrome_num_dict = {'chr1':1, 'chr2':2, 'chr3':3, 'chr4':4, 'chr5':5, 'chr6':6, 'chr7':7, 'chr8':8, 'chr9':9, 'chr10':10, 'chr11':11, 'chr12':12, 'chr13':13, 'chr14':14, 'chr15':15, 'chr16':16, 'chr17':17, 'chr18':18, 'chr19':19, 'chr20':20, 'chr21':21, 'chr22':22, 'chrX':23, 'chrY':24}
other_source_labels_dict = {'Benign':1,'LikelyPathogenic':8,'Pathogenic':9,'VUS_I':7,'VUS_II':6,'VUS_III':5,'VUS_V':4,'VUS_VI':3,'VUS_VII':2,'':10}
splicing_type_dict = {'':1, 'CERTAIN':2, 'POSSIBLE':3}
conservation_dict = {'':0,'Conserved':1,'Not available':2, 'MildlyConserved':3, 'NotConserved':4}
boolean_dict = {'true':1,'false':0}
damaging_preds_dict = {'Tolerated':0, 'PolymorphismAutomatic':1, 'DiseaseCausing':2, 'ProbablyDamaging':3, 'Unknown':4, 'Neutral':5, 'High':6, 'Benign':7, 'Deleterious':8, 'Medium':9, 'Low':10, 'Polymorphism':11, 'DiseaseCausingAutomatic':12, 'Damaging':13, 'PossiblyDamaging':14,'':15}

def get_uni_bi_basetype_vector(pattern):
	return [pattern.count('A'),pattern.count('T'),pattern.count('C'),pattern.count('G'),pattern.count('AA'),pattern.count('TA'),pattern.count('CA'),pattern.count('GA'),pattern.count('AT'),pattern.count('TT'),pattern.count('CT'),pattern.count('GT'),pattern.count('AC'),pattern.count('TC'),pattern.count('CC'),pattern.count('GC'),pattern.count('AG'),pattern.count('TG'),pattern.count('CG'),pattern.count('GG')]

def list_str_to_int(str_list):
	return [int(x) for x in str_list]

def rescale(x,min,max):
	return float(2*x - max - min)/float(max - min)

with open('test.csv') as ip:
	reader = csv.reader(ip, delimiter=',')
	features_dict = {}
	for row in reader:
		row_features = []
		row_features.append(int(row[0]))
		row_features.append(chrome_num_dict[row[1]])
		# row_features.append(rescale(int(row[2]),13905,249212111))
		row_features = row_features + get_uni_bi_basetype_vector(row[3])
		row_features = row_features + get_uni_bi_basetype_vector(row[4])
		row_features.append(int(row[5]))
		row_features.append(int(row[6]))
		# if row[8] != '':
		# 	row_features.append(rescale(float(row[8]),0,1))
		# else:
		# 	row_features.append(0.03999067402)

		# if row[9] != '':
		# 	row_features.append(rescale(float(row[9]),0,1))
		# else:
		# 	row_features.append(0.04150059397)


		# if row[10] != '':
		# 	row_features.append(rescale(float(row[10]),0,1))
		# else:
		# 	row_features.append(0.03692720187)

		# if row[11] != '':
		# 	row_features.append(rescale(float(row[11]),0,1))
		# else:
		# 	row_features.append(0.04273295511)

		#11
		row_features += [other_source_labels_dict[x] for x in row[12:25]]
		row_features.append(conservation_dict[row[26]])
		row_features.append(conservation_dict[row[27]])
		row_features.append(conservation_dict[row[28]])
		# if row[29] != '':
		# 	row_features.append(rescale(float(row[29]),0,1))
		# else:
		# 	row_features.append(0.0998810938)
			
		row_features.append(boolean_dict[row[30]])
		row_features.append(boolean_dict[row[31]])
		row_features.append(boolean_dict[row[32]])
		row_features += list_str_to_int(row[33:54])
		row_features += [damaging_preds_dict[x] for x in row[54:61]]
		if row[61] != '':
			row_features.append(int(row[61]))
		else:
			row_features.append(2)
		row_features.append(boolean_dict[row[62]])
		row_features.append(boolean_dict[row[63]])
		row_features.append(boolean_dict[row[64]])
		row_features.append(boolean_dict[row[65]])
		# row_features.append(int(row[66]))
		
		# print row[2]
		# # exit()

		# train_dict[row[0]] = {"features":[],"label":""}
		# train_dict[row[0]]["features"] = row[1:66]
		# train_dict[row[0]]["label"] = row[66]
		features_dict[row[0]] = row_features	

# print len(features_dict['84091'])

with open('test_feature_95.pkl','w') as op:
	pickle.dump(features_dict,op)
# z = []
# ref_list_dict = {}
# label_list = []
# for x in train_dict.keys():
# 	# try:
# 	# 	ref_list_dict[train_dict[x]["features"][3]] = ref_list_dict[train_dict[x]["features"][3]] + 1
# 	# except:
# 	# 	ref_list_dict[train_dict[x]["features"][3]] = 1

# 	l_list = train_dict[x]["features"][53:60]
# 	l_list = [x for x in train_dict[x]["features"][53:60] if x != ""]
# 	# print l_list
# 	label_list = label_list + l_list
# 	# label_list.append(train_dict[x]["features"][29])
# 	# ref_list.append(train_dict[x]["features"][2])
# 	# z.append(train_dict[x]["label"])
# # print label_list
# print set(label_list)
# # print set(ref_list), len(set(ref_list))
# # print len(ref_list_dict.keys())
# # print sort_dict(ref_list_dict)
