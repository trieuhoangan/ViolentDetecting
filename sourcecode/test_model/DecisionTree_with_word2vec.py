import pymysql.cursors
import numpy as np


from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from gensim import models
from gensim.models import KeyedVectors

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

np.seterr(divide='ignore', invalid='ignore')
import sys
sys.path.append("..")
from customlib import vectorizeText,accessMysql,check_params

print("load data from mysql")
all_text = accessMysql.getContentList("select * from sentences")

index_link = "data_index.txt"
f = open(index_link,'r')
train_data_begin_index = int(f.readline().replace('\n',''))
train_data_end_index = int(f.readline().replace('\n',''))
test_data_begin_index = int(f.readline().replace('\n',''))
test_data_end_index = int(f.readline().replace('\n',''))
numbertestcase = test_data_end_index

# print("load data from mysql")
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
texts.extend(sentence_label_0)
texts.extend(sentence_label_1)
texts.extend(sentence_label_2)
labels.extend(label_type_0)
labels.extend(label_type_1)
labels.extend(label_type_2)
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

print('vectorize sentences')
model_link = '../models_bin/wiki.vi.model.bin'
word2vec_model = KeyedVectors.load_word2vec_format(model_link, binary=True)

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

print('train model')
scores = ['precision', 'recall']
params = [{'random_state':[5,6,7,8,9]}]
# for score in scores:
#     model = GridSearchCV(DecisionTreeClassifier(),params,cv=5,scoring='%s_macro' % score).fit(vectors, labels)
model = DecisionTreeClassifier(random_state=6).fit(vectors, labels)
test_result = model.predict(test_vectors)
print("test result:")
# print(test_result)
print("--------------------------------------------------------------")
print("test examples:")
# print(test_labels)
print(" best params : ")
# print(model.best_params_)
accurracy = "accuracy score : "+ str(accuracy_score(test_labels,test_result))
f1_score_macro = "f1 score ave macro : "+str(f1_score(test_labels,test_result,average='macro'))
f1_score_micro = "f1 score ave micro : "+str(f1_score(test_labels,test_result,average='micro'))
f1_score_weighted = "f1 score ave weighted : "+str(f1_score(test_labels,test_result,average='weighted'))
precision_score_macro = "precision score ave macro : " + str(precision_score(test_labels,test_result,average='macro'))
precision_score_micro = "precision score ave micro : " + str(precision_score(test_labels,test_result,average='micro'))
precision_score_weighted = "precision score ave weighted : " + str(precision_score(test_labels,test_result,average='weighted'))
# print("--------------------------------------------------------------")
print(accurracy)
print(f1_score_macro)
print(f1_score_micro)
print(f1_score_weighted)
print(precision_score_macro)
print(precision_score_micro)
print(precision_score_weighted)

print(check_params.check_accuracy_of_label(numbertestcase,0,test_result))
print(check_params.check_accuracy_of_label(numbertestcase,1,test_result))
print(check_params.check_accuracy_of_label(numbertestcase,2,test_result))
print("--------------------------------------------------------------------------------------------------------------------")
# tets_texts = accessMysql.getContentList("select * from newspaper where id = 1")
# words = vectorizeText.split_list(tets_texts)
# for word in words:
#     for single in word:
#         print(word2vec_model.word_vec(single))


from joblib import dump
model_name = '../models_bin/RandomForest_with_word2vec.joblib'
dump(model,model_name)

