import numpy as np
from joblib import dump,load
from sklearn.ensemble import VotingClassifier
from gensim.models import KeyedVectors
from sklearn.model_selection import GridSearchCV
import sys
sys.path.append("..")
from customlib import vectorizeText,accessMysql,check_params
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
LogisticModel_tfidf_link = '../models_bin/losgisticRegression_with_tfidf.joblib'
LogisticModel_word2vec_link = '../models_bin/losgisticRegression_with_word2vec.joblib'
RandomForest_tdidf_link = '../models_bin/RandomForest_with_tfidf.joblib'
RandomForest_word2vec_link = '../models_bin/RandomForest_with_word2vec.joblib'
# Gaussian_tfidf_link = '../models_bin/gaussian_with_tfidf.joblib'
# Gaussian_word2vec_link = '../models_bin/gaussian_with_word2vec.joblib'
SVC_tfidf_link = '../models_bin/SVC_with_tfidf.joblib'
SVC_word2vec_link = '../models_bin/SVC_with_word2vec.joblib'
word2vec_vectorize_model_link = '../models_bin/wiki.vi.model.bin'
tfidf_vectorize_model_link = '../models_bin/tfidf_model.joblib'
tf = load(tfidf_vectorize_model_link)
word2vec_model = KeyedVectors.load_word2vec_format(word2vec_vectorize_model_link, binary=True)
word2vec_model_links = [LogisticModel_word2vec_link,SVC_word2vec_link,RandomForest_word2vec_link]
tfidf_model_links = [LogisticModel_tfidf_link,SVC_tfidf_link,RandomForest_tdidf_link]

word2vec_models = []
Tfidf_models=[]
RandomForestModelw2v = RandomForestClassifier(n_estimators=50, random_state=4,max_leaf_nodes=48,max_depth=7)
LogisticRegressionModelw2v = LogisticRegression(penalty='l2',solver='lbfgs',max_iter=480,multi_class='ovr',random_state=0)
SVCModelw2v = SVC(kernel='rbf',gamma='scale',decision_function_shape='ovo',C=8.088,probability=True)
RandomForestModeltfidf = RandomForestClassifier(n_estimators=40, random_state=7,max_leaf_nodes=10,max_depth=7)
LogisticRegressionModeltfidf = LogisticRegression(penalty='l1',solver='saga',multi_class='ovr',max_iter=280)
SVCModeltfidf = SVC(kernel='rbf',gamma='scale',C=8.5,decision_function_shape='ovo',probability=True)

word2vec_models.append(LogisticRegressionModelw2v)
word2vec_models.append(SVCModelw2v)
word2vec_models.append(RandomForestModelw2v)
Tfidf_models.append(LogisticRegressionModeltfidf)
Tfidf_models.append(SVCModeltfidf)
Tfidf_models.append(RandomForestModeltfidf)
# for link in word2vec_model_links:
#     with open(link,'rb') as f:
#         print(link)
#         model = load(link)
#         # print(model.predict(test_vec))
#         word2vec_models.append(model)
# for link in tfidf_model_links:
#     with open(link,'rb') as f:
#         print(link)
#         model = load(link)
#         # print(model.predict(testvector))
#         Tfidf_models.append(model)

print('loading test data')
index_link = "data_index.txt"
f = open(index_link,'r')
train_data_begin_index = int(f.readline().replace('\n',''))
train_data_end_index = int(f.readline().replace('\n',''))
test_data_begin_index = int(f.readline().replace('\n',''))
test_data_end_index = int(f.readline().replace('\n',''))
numbertestcase = test_data_end_index

print("load data from mysql")
all_text = accessMysql.getContentList("select * from sentences")
sentence_label_0 = accessMysql.getContentListWithLimit("select * from sentences where label = 0 limit %s,%s",train_data_begin_index,train_data_end_index)
label_type_0 = accessMysql.getLabelListWithLimit("select * from sentences where label = 0  limit %s,%s",train_data_begin_index,train_data_end_index)
test_data_label_type_0 = accessMysql.getContentListWithLimit("select * from sentences where label = 0  limit %s,%s",test_data_begin_index,test_data_end_index)
test_label_type_0 = accessMysql.getLabelListWithLimit("select * from sentences where label = 0  limit %s,%s",test_data_begin_index,test_data_end_index)

sentence_label_1 = accessMysql.getContentListWithLimit("select * from sentences where label = 1 limit %s,%s",train_data_begin_index,train_data_end_index)
# sentence_label_1_part2 = accessMysql.getContentListWithLimit("select * from sentences where label = 1 limit %s,%s",700,1100)
label_type_1 = accessMysql.getLabelListWithLimit("select * from sentences where label = 1  limit %s,%s",train_data_begin_index,train_data_end_index)
# label_type_1_part2 = accessMysql.getLabelListWithLimit("select * from sentences where label = 1  limit %s,%s",700,1100)
test_data_label_type_1 = accessMysql.getContentListWithLimit("select * from sentences where label = 1  limit %s,%s",test_data_begin_index,test_data_end_index)
test_label_type_1 = accessMysql.getLabelListWithLimit("select * from sentences where label = 1  limit %s,%s",test_data_begin_index,test_data_end_index)

sentence_label_2 = accessMysql.getContentListWithLimit("select * from sentences where label = 2 limit %s,%s",train_data_begin_index,train_data_end_index)
# sentence_label_2_part2 = accessMysql.getContentListWithLimit("select * from sentences where label = 2 limit %s,%s",700,1100)
label_type_2 = accessMysql.getLabelListWithLimit("select * from sentences where label = 2  limit %s,%s",train_data_begin_index,train_data_end_index)
# label_type_2_part2 = accessMysql.getLabelListWithLimit("select * from sentences where label = 2  limit %s,%s",700,1100)
test_data_label_type_2 = accessMysql.getContentListWithLimit("select * from sentences where label = 2  limit %s,%s",test_data_begin_index,test_data_end_index)
test_label_type_2 = accessMysql.getLabelListWithLimit("select * from sentences where label = 2  limit %s,%s",test_data_begin_index,test_data_end_index)
texts = []
labels = []
test_data = []
test_labels = []
# print(label_type_0)
texts.extend(sentence_label_0)
texts.extend(sentence_label_1)
texts.extend(sentence_label_2)
# texts.extend(sentence_label_1_part2)
# texts.extend(sentence_label_2_part2)
labels.extend(label_type_0)
labels.extend(label_type_1)
labels.extend(label_type_2)
# labels.extend(label_type_1_part2)
# labels.extend(label_type_2_part2)
test_data.extend(test_data_label_type_0)
test_data.extend(test_data_label_type_1)
test_data.extend(test_data_label_type_2)
test_labels.extend(test_label_type_0)
test_labels.extend(test_label_type_1)
test_labels.extend(test_label_type_2)
datas = []
for text in texts:
    datas.append(text.replace('_',' '))
test_cases = []
for test_case in test_data:
    test_cases.append(test_case.replace('_',' '))
tokens = vectorizeText.split_list(datas)

vectors = [] 
for token in tokens:
    vectors.append(vectorizeText.sent_vectorize(token,word2vec_model))
vectors = np.nan_to_num(vectors)

print("create testcases")
test_sequence_list = vectorizeText.split_list(test_cases)
test_vectors = []
for test_sequence in test_sequence_list:
    test_vectors.append(vectorizeText.sent_vectorize(test_sequence,word2vec_model))
test_vectors = np.nan_to_num(test_vectors)

tf_vector = tf.transform(texts)
tf_test_vector = tf.transform(test_data)

print('predicting')
ensemble_word2vec = VotingClassifier(estimators=[('lr',word2vec_models[0]),('svm',word2vec_models[1]),('rf',word2vec_models[2])],voting='hard')
ensemble_word2vec.fit(vectors, labels)
ensemble_tfidf = VotingClassifier(estimators=[('lr',LogisticRegressionModeltfidf),('svc',SVCModeltfidf),('rf',RandomForestModeltfidf)],voting='hard')
ensemble_tfidf.fit(tf_vector,labels)
w2v_test_result = ensemble_word2vec.predict(test_vectors)
tfidf_test_result = ensemble_tfidf.predict(tf_test_vector)
print(accuracy_score(test_labels,w2v_test_result))
print(accuracy_score(test_labels,tfidf_test_result))
print(f1_score(test_labels,w2v_test_result,average=None))
print(f1_score(test_labels,tfidf_test_result,average=None))



w_model_name = '../models_bin/ensemble_hardvoting_with_word2vec.joblib'
dump(ensemble_word2vec,w_model_name) 

tf_model_name = '../models_bin/ensemble_hardvoting_with_tf.joblib'
dump(ensemble_tfidf,tf_model_name)