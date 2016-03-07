import pickle
import numpy
import pandas as pd
from pandas import DataFrame
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

def build_data_frame(path,classification,i=0):
    rows = []
    index = []
    
    data_dict = pickle.load(open(path,'r'))
    for rt_id,email in data_dict.iteritems():
    	rows.append({'text': email, 'class': classification})
    	index.append(i)
    	i = i + 1
	
	data_frame = DataFrame(rows, index=index)
    return data_frame

def append_data_frame(path,data_frame,classification):
    data_frame2 = build_data_frame(path,classification,len(data_frame.index))
    return pd.concat([data_frame,data_frame2])

data = build_data_frame('positivedata.pkl','pos')
data = append_data_frame('negativedata.pkl',data,'neg')

data = data.reindex(numpy.random.permutation(data.index))

pos_data_frame = data.loc[data['class'] == 'pos']
train_pos_indices = [i for i in xrange(0,int(0.8*len(pos_data_frame.index)))]
train_pos_data_frame = pos_data_frame.iloc[train_pos_indices]
test_pos_indices = [i for i in xrange(int(0.8*len(pos_data_frame.index)),len(pos_data_frame.index))]
test_pos_data_frame = pos_data_frame.iloc[test_pos_indices]
print len(pos_data_frame.index), len(train_pos_data_frame.index), len(test_pos_data_frame.index)

neg_data_frame = data.loc[data['class'] == 'neg']
train_neg_indices = [i for i in xrange(0,int(0.8*len(neg_data_frame.index)))]
train_neg_data_frame = neg_data_frame.iloc[train_neg_indices]
test_neg_indices = [i for i in xrange(int(0.8*len(neg_data_frame.index)),len(neg_data_frame.index))]
test_neg_data_frame = neg_data_frame.iloc[test_neg_indices]
print len(neg_data_frame.index),len(train_neg_data_frame.index), len(test_neg_data_frame.index)

train_data_frame = pd.concat([train_pos_data_frame,train_neg_data_frame])
train_data_frame = train_data_frame.reindex(numpy.random.permutation(train_data_frame.index))

test_data_frame = pd.concat([test_pos_data_frame,test_neg_data_frame])
test_data_frame = test_data_frame.reindex(numpy.random.permutation(test_data_frame.index))

print len(train_data_frame.index), len(test_data_frame)

# count_vectorizer = CountVectorizer()
# counts = count_vectorizer.fit_transform(data['text'].values)

# classifier = MultinomialNB()
# targets = data['class'].values
# classifier.fit(counts, targets)

from sklearn.pipeline import Pipeline

########################################### Bernoulli NB Classifier #####################################
# from sklearn.naive_bayes import BernoulliNB

# pipeline = Pipeline([
#     ('count_vectorizer',   CountVectorizer(ngram_range=(1, 2))),
#     ('classifier',         BernoulliNB(binarize=0.0)) ])
# ('Total emails classified:', 3657)
# ('Score:', 0.75745148358514258)
# Confusion matrix:
# [[1277  661]
#  [ 219 1500]]


########################################### Multinomial NB Classifier #####################################
# from sklearn.feature_extraction.text import TfidfTransformer

# pipeline = Pipeline([
#     ('count_vectorizer',   CountVectorizer(ngram_range=(1,  2))),
#     ('tfidf_transformer',  TfidfTransformer()),
#     ('classifier',         MultinomialNB())
# ])

# ('Total emails classified:', 4957)
# ('Score:', 0.83800565402130178)
# Confusion matrix:
# [[3773    0]
#  [ 690  494]]

########################################### SGD Classifier #####################################
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfTransformer
pipeline = Pipeline([
    ('count_vectorizer',   CountVectorizer(ngram_range=(1,  2))),
    ('tfidf_transformer',  TfidfTransformer()),
    ('classifier',         SGDClassifier())
])
# ('Total emails classified:', 4957)
# ('Score:', 0.98453550033595072)
# Confusion matrix:
# [[3762   11]
#  [  65 1119]]


from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, f1_score

k_fold = KFold(n=len(train_data_frame), n_folds=6)
scores = []
confusion = numpy.array([[0, 0], [0, 0]])
i = 0
for train_indices, test_indices in k_fold:
    train_text = train_data_frame.iloc[train_indices]['text'].values
    train_y = train_data_frame.iloc[train_indices]['class'].values

    test_text = train_data_frame.iloc[test_indices]['text'].values
    test_y = train_data_frame.iloc[test_indices]['class'].values

    pipeline.fit(train_text, train_y)
    predictions = pipeline.predict(test_text)

    print 'fold ' + str(i)
    print confusion_matrix(test_y, predictions)
    print f1_score(test_y, predictions,labels=None,pos_label=None, average='weighted')
    
    confusion += confusion_matrix(test_y, predictions)
    score = f1_score(test_y, predictions,labels=None,pos_label=None, average='weighted')
    scores.append(score)
    i += 1

# print('Total emails classified:', len(data))
# print('Score:', sum(scores)/len(scores))
print('Confusion matrix:')
print(confusion)

# from sklearn.externals import joblib
# joblib.dump(pipeline, 'pos_neg_classifier_model.pkl', compress=9)

# # Then to load it back in 




# print 'Hold Out ---'
# trained_pipeline = joblib.load('pos_neg_classifier_model.pkl')

# test_text = test_data_frame['text'].values
# test_y = test_data_frame['class'].values

# predictions = trained_pipeline.predict(test_text)

# confusion = confusion_matrix(test_y, predictions)
# score = f1_score(test_y, predictions,labels=None,pos_label=None, average='weighted')

# print('Total emails held back:', len(test_data_frame))
# print('Score:', score)
# print('Confusion matrix:')
# print(confusion)