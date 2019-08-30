import pymysql.cursors
import numpy as np

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
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
import re
clean_spe_char = re.compile('[\\~#%&*{}/:<>?,.;|"]')
for text in texts:
    # text = re.sub(clean_spe_char,' ',text)
    # text = text.replace("'",' ')
    text = text.replace('_',' ')
    datas.append(text)
test_cases = []
for test_case in test_data:
    # test_case=  re.sub(clean_spe_char,' ',test_case)
    # test_case = test_case.replace("'",' ')
    test_cases.append(test_case.replace('_',' '))
tokens = vectorizeText.split_list(datas)

print('vectorize sentences')
model_link = '../models_bin/wiki.vi.model.bin'
word2vec_model = KeyedVectors.load_word2vec_format(model_link, binary=True)

vectors = [] 
for token in tokens:
    vectors.append(vectorizeText.sent_vectorize(token,word2vec_model))
vectors = np.nan_to_num(vectors)
test_sequence_list = vectorizeText.split_list(test_cases)
# print(test_sequence_list)
test_vectors = []
for test_sequence in test_sequence_list:
    test_vectors.append(vectorizeText.sent_vectorize(test_sequence,word2vec_model))
test_vectors = np.nan_to_num(test_vectors)
print('train model')
scores = ['precision', 'recall']
params = [{'C':[8.07,8.08]}]


# for score in scores:
#     model = GridSearchCV(SVC(kernel='rbf',gamma='scale',decision_function_shape='ovo'),params,cv=5,scoring='%s_macro' % score).fit(vectors,labels)
model = SVC(kernel='rbf',gamma='scale',decision_function_shape='ovo',C=8.088).fit(vectors,labels)
test_result = model.predict(test_vectors)
    # print(accuracy_score(test_labels,test_result))
print("test result:")
# print(test_result)
print("--------------------------------------------------------------")
print("test examples:")
# print(test_labels)

# tets_texts = accessMysql.getContentList("select * from newspaper where id = 1")
# words = vectorizeText.split_list(tets_texts)
# for word in words:
#     for single in word:
#         print(word2vec_model.word_vec(single))
accurracy = "accuracy score : "+ str(accuracy_score(test_labels,test_result))
f1_score_macro = "f1 score ave macro : "+str(f1_score(test_labels,test_result,average='macro'))
f1_score_micro = "f1 score ave micro : "+str(f1_score(test_labels,test_result,average='micro'))
f1_score_weighted = "f1 score ave weighted : "+str(f1_score(test_labels,test_result,average='weighted'))
precision_score_macro = "precision score ave macro : " + str(precision_score(test_labels,test_result,average='macro'))
precision_score_micro = "precision score ave micro : " + str(precision_score(test_labels,test_result,average='micro'))
precision_score_weighted = "precision score ave weighted : " + str(precision_score(test_labels,test_result,average='weighted'))
# print(model.best_params_)
print("--------------------------------------------------------------")
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


from joblib import dump
model_name = '../models_bin/SVC_with_word2vec.joblib'
dump(model,model_name)


# link_exp = "experiments/svm_w_exp_start%s_length_%s.txt"%(str(train_data_begin_index),(str((train_data_end_index)*3)))
# f = open(link_exp,"w+",encoding="utf-8")
# Cs=[8.07,8.08,8.09]
# kernels=[ "linear","poly","rbf","sigmoid"]
# decision_function_shape=['ovo','ovr']
# for C in Cs:
#     for shape_f in decision_function_shape:
#         for kernel in kernels:

#             model = SVC(kernel=kernel,gamma='scale',decision_function_shape=shape_f,C=C)
#             model.fit(vectors,labels)
#             test_result = model.predict(test_vectors)
#             accurracy = "accuracy score : "+ str(accuracy_score(test_labels,test_result))
#             f1_score_macro = "f1 score ave macro : "+str(f1_score(test_labels,test_result,average='macro'))
#             f1_score_micro = "f1 score ave micro : "+str(f1_score(test_labels,test_result,average='micro'))
#             f1_score_weighted = "f1 score ave weighted : "+str(f1_score(test_labels,test_result,average='weighted'))
#             precision_score_macro = "precision score ave macro : " + str(precision_score(test_labels,test_result,average='macro'))
#             precision_score_micro = "precision score ave micro : " + str(precision_score(test_labels,test_result,average='micro'))
#             precision_score_weighted = "precision score ave weighted : " + str(precision_score(test_labels,test_result,average='weighted'))
#             params ="kernel : "+kernel+", decision_function_shape : "+str(shape_f)+", C : "+str(C)+" , "+accurracy+" , "+f1_score_macro+"\n"
#             f.write(params)

# f.close