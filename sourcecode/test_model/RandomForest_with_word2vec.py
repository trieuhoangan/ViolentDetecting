import pymysql.cursors
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from gensim import models
from gensim.models import KeyedVectors

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

# from customlib import vectorizeText
np.seterr(divide='ignore', invalid='ignore')
import sys
sys.path.append("..")
from customlib import vectorizeText,accessMysql,check_params



index_link = "data_index.txt"
f = open(index_link,'r')
train_data_begin_index = int(f.readline().replace('\n',''))
train_data_end_index = int(f.readline().replace('\n',''))
test_data_begin_index = int(f.readline().replace('\n',''))
test_data_end_index = int(f.readline().replace('\n',''))
numbertestcase = test_data_end_index

# print("load data from mysql")
print("load data from mysql")
all_text = accessMysql.getContentList("select * from sentences")
sentence_label_0 = accessMysql.getContentListWithLimit("select * from sentences where label = 0 and istrain = 1 limit %s,%s",train_data_begin_index,train_data_end_index)
label_type_0 = accessMysql.getLabelListWithLimit("select * from sentences where label = 0 and istrain = 1  limit %s,%s",train_data_begin_index,train_data_end_index)
test_data_label_type_0 = accessMysql.getContentListWithLimit("select * from sentences where label = 0 and istrain = 1  limit %s,%s",test_data_begin_index,test_data_end_index)
test_label_type_0 = accessMysql.getLabelListWithLimit("select * from sentences where label = 0 and istrain = 1  limit %s,%s",test_data_begin_index,test_data_end_index)

sentence_label_1 = accessMysql.getContentListWithLimit("select * from sentences where label = 1 and istrain = 1 limit %s,%s",train_data_begin_index,train_data_end_index)
# sentence_label_1_part2 = accessMysql.getContentListWithLimit("select * from sentences where label = 1 and istrain = 1 limit %s,%s",700,1100)
label_type_1 = accessMysql.getLabelListWithLimit("select * from sentences where label = 1 and istrain = 1  limit %s,%s",train_data_begin_index,train_data_end_index)
# label_type_1_part2 = accessMysql.getLabelListWithLimit("select * from sentences where label = 1 and istrain = 1  limit %s,%s",700,1100)
test_data_label_type_1 = accessMysql.getContentListWithLimit("select * from sentences where label = 1 and istrain = 1  limit %s,%s",test_data_begin_index,test_data_end_index)
test_label_type_1 = accessMysql.getLabelListWithLimit("select * from sentences where label = 1 and istrain = 1  limit %s,%s",test_data_begin_index,test_data_end_index)

sentence_label_2 = accessMysql.getContentListWithLimit("select * from sentences where label = 2 and istrain = 1 limit %s,%s",train_data_begin_index,train_data_end_index)
# sentence_label_2_part2 = accessMysql.getContentListWithLimit("select * from sentences where label = 2 and istrain = 1 limit %s,%s",700,1100)
label_type_2 = accessMysql.getLabelListWithLimit("select * from sentences where label = 2 and istrain = 1  limit %s,%s",train_data_begin_index,train_data_end_index)
# label_type_2_part2 = accessMysql.getLabelListWithLimit("select * from sentences where label = 2 and istrain = 1  limit %s,%s",700,1100)
test_data_label_type_2 = accessMysql.getContentListWithLimit("select * from sentences where label = 2 and istrain = 1  limit %s,%s",test_data_begin_index,test_data_end_index)
test_label_type_2 = accessMysql.getLabelListWithLimit("select * from sentences where label = 2 and istrain = 1  limit %s,%s",test_data_begin_index,test_data_end_index)
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
# scores = ['precision', 'recall']
# params = [{'max_depth':[7,8,9],'n_estimators':[40,45,50],'max_leaf_nodes':[46,48,50],'random_state':[3,4,5]}]
# for score in scores:
# RandomForestModel = GridSearchCV(RandomForestClassifier(),params,cv=5,scoring='%s_macro' % score).fit(vectors, labels)
RandomForestModel = RandomForestClassifier(n_estimators=50, random_state=4,max_leaf_nodes=48,max_depth=7).fit(vectors, labels)
test_result = RandomForestModel.predict(test_vectors)
print("test result:")
print("--------------------------------------------------------------")
print("test examples:")
# print(test_labels)
# print(" best params : ")
# print(RandomForestModel.best_params_)

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
dump(RandomForestModel,model_name)



#for experiment only
# link_exp = "experiments/rf_w_exp_start%s_length_%s.txt"%(str(train_data_begin_index),(str((train_data_end_index)*3)))
# f = open(link_exp,"w+",encoding="utf-8")
# max_leaf_nodes=[46,48,50]
# max_depth=[7,8,9]
# n_estimators=[40,45,50]
# random_state=[0,2,4,6,8]

# model = RandomForestClassifier()
# for leaf in max_leaf_nodes:
#     for dept in max_depth:
#         for state in random_state:
#             for n in n_estimators:
#                 model = RandomForestClassifier(n_estimators=n,random_state=state,max_depth=dept,max_leaf_nodes=leaf)
#                 model.fit(vectors,labels)
#                 test_result = model.predict(test_vectors)
#                 accurracy = "accuracy score : "+ str(accuracy_score(test_labels,test_result))
#                 f1_score_macro = "f1 score ave macro : "+str(f1_score(test_labels,test_result,average='macro'))
#                 f1_score_micro = "f1 score ave micro : "+str(f1_score(test_labels,test_result,average='micro'))
#                 f1_score_weighted = "f1 score ave weighted : "+str(f1_score(test_labels,test_result,average='weighted'))
#                 precision_score_macro = "precision score ave macro : " + str(precision_score(test_labels,test_result,average='macro'))
#                 precision_score_micro = "precision score ave micro : " + str(precision_score(test_labels,test_result,average='micro'))
#                 precision_score_weighted = "precision score ave weighted : " + str(precision_score(test_labels,test_result,average='weighted'))
#                 params = "max_leaf_nodes : "+str(leaf)+" max_depth : "+str(dept)+" n_estimators : "+str(n)+" random_state : "+str(state)+" "+accurracy+" "+f1_score_macro+"\n"
#                 f.write(params)

# f.close