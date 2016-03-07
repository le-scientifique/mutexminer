import csv
import pickle
import operator
train_dict = {}
def sort_dict(x):
	# x = {1: 2, 3: 4, 4: 3, 2: 1, 0: 0}
	sorted_x = sorted(x.items(), key=operator.itemgetter(1),reverse=True)
	return sorted_x

with open('train.csv') as ip:
	reader = csv.reader(ip, delimiter=',')
	for row in reader:
		# print row[0]
		# print type(row[0])
		# print row[66]
		# print row[1:66]
		train_dict[row[0]] = {"features":[],"label":""}
		train_dict[row[0]]["features"] = row[1:66]
		train_dict[row[0]]["label"] = row[66]
z = []
ref_list_dict = {}
label_list = []
for x in train_dict.keys():
	try:
		#ref_list_dict[train_dict[x]["features"][60]] = ref_list_dict[train_dict[x]["features"][60]] + 1
		ref_list_dict[train_dict[x]["label"]] = ref_list_dict[train_dict[x]["label"]] + 1
	except:
		ref_list_dict[train_dict[x]["label"]] = 1

	# l_list = train_dict[x]["features"][53:60]
	# l_list = [x for x in train_dict[x]["features"][53:60] if x != ""]
	# # print l_list
	# label_list = label_list + l_list


	# label_list.append(train_dict[x]["features"][61])
	# ref_list.append(train_dict[x]["features"][2])
	# z.append(train_dict[x]["label"])
# print label_list
# print sorted(set(label_list))
# print set(ref_list), len(set(ref_list))
# print len(ref_list_dict.keys())
print sort_dict(ref_list_dict)
