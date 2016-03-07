import numpy as np
from sklearn import linear_model
import pickle
from sklearn.externals import joblib

data_dict = pickle.load(open('feature_levenshtein_conts.pkl'))
data_list = data_dict.values()
data_matrix = np.array(data_list)
#print X.shape
X = data_matrix[:,:59] 
print X.shape
Y = data_matrix[:,59:]
print Y.shape

from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.multiclass import OutputCodeClassifier

pipeline = Pipeline([
    ('classifier',  MLPClassifier(hidden_layer_sizes=100,alpha=0.01,activation='relu'))
])


# pipeline = Pipeline([
#     ('classifier',  tree.DecisionTreeClassifier())
# ])

# pipeline = Pipeline([
#     ('classifier',  RandomForestClassifier(n_estimators=10))
# ])



# pipeline = Pipeline([
#     ('classifier',  GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0))
# ])

# pipeline = Pipeline([
#     ('classifier',  AdaBoostClassifier(
#     DecisionTreeClassifier(max_depth=2),
#     n_estimators=600,
#     learning_rate=1.5,
#     algorithm="SAMME"))
# ])

# pipeline = Pipeline([
#     ('classifier',OutputCodeClassifier(LinearSVC(random_state=0),code_size=2, random_state=0))
# ])


# pipeline = Pipeline([
#     ('classifier', OneVsRestClassifier(LinearSVC(random_state=0)))
# ])

# pipeline = Pipeline([
#     ('classifier', MultinomialNB())
# ])

# pipeline = Pipeline([
#     ('classifier', GaussianNB())
# ])

from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD

# svd = TruncatedSVD(n_components=10)
# normalizer = Normalizer(copy=False)
# lsa = make_pipeline(svd, normalizer)
# X = lsa.fit_transform(X)

no_of_rows,_ = X.shape

k_fold = KFold(n=no_of_rows, n_folds=6)
scores = []
confusion = np.zeros((9,9),dtype=np.int)# numpy.array([[0, 0], [0, 0]])
i = 0
for train_indices, test_indices in k_fold:
	i = i + 1
	train_ftrs = X[train_indices]# X.iloc[train_indices]['text'].values
	train_y = Y[train_indices]
	# print train_ftrs.shape
	# print train_y.shape
	# train_y = train_data_frame.iloc[train_indices]['class'].values

	test_ftrs = X[test_indices] 
	test_y = Y[test_indices] 

	pipeline.fit(train_ftrs, train_y)
	predictions = pipeline.predict(test_ftrs)
	
	joblib.dump(pipeline, 'mutex_classifier_mlp_lev_cont' + str(i) + 'model.pkl', compress=9)
	print test_y, predictions
	print 'fold ' + str(i)
	print confusion_matrix(test_y, predictions,labels=[1,2,3,4,5,6,7,8,9])
	print "Precision: ", precision_score(test_y, predictions, average='micro')
	print "Recall: ", recall_score(test_y, predictions, average='micro')
	# # prin
	# confusion += confusion_matrix(test_y, predictions,labels=[0,1,2,3,4,5,6,7,8,9])
	# score = f1_score(test_y, predictions,labels=[0,1,2,3,4,5,6,7,8,9],pos_label=None, average='weighted')
	# scores.append(score)
	# i += 1

print('Confusion matrix:')
print(confusion)

# mlp lev cont 