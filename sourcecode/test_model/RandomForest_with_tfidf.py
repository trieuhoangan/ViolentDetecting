import re
import sys
sys.path.append("..")

from sklearn.model_selection import GridSearchCV
from customlib import vectorizeText,accessMysql,check_params
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import numpy as np
from joblib import dump, load



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
print('preprocessing data')
valid_file_name_character = re.compile('[\\~#%&*{}/:<>?|\"-]')

words = []
for text in all_text:
    text = re.sub(valid_file_name_character,'',text)
    words.append(text.split(' '))
    
stop_word = ['.',',',';','!','@','#','-','>','(',')','/']
#create dictionary 
tfidf_vectorize_model_link = '../models_bin/tfidf_model.joblib'
tf = load(tfidf_vectorize_model_link)
vectors = tf.transform(texts)
print('vector shape')
print(vectors.shape)
print(len(labels))
# for text in texts:
#     text = re.sub(valid_file_name_character,'',text)
#     vectors.append(tf.transform(text.split('.')))
#     # print(vectors)
print('training model ')
test_vectors = tf.transform(test_data)
scores = ['precision', 'recall']
params = [{'max_depth':[7],'n_estimators':[40],'random_state':[7],'max_leaf_nodes':[10],'bootstrap':[True]}]
RandomForestModel = RandomForestClassifier(n_estimators=40, random_state=7,max_leaf_nodes=10,max_depth=7).fit(vectors, labels)
# for score in scores:
#     RandomForestModel = GridSearchCV(RandomForestClassifier(),params,cv=5,scoring='%s_macro' % score).fit(vectors, labels)
# print('testing')

test_result = RandomForestModel.predict(test_vectors)
print(accuracy_score(test_labels,test_result))
print("test result:")
# print(test_result)
print("--------------------------------------------------------------")
print("test ex amples:")
# print(test_labels)
# print(RandomForestModel.best_params_)
print(check_params.check_accuracy_of_label(numbertestcase,0,test_result))
print(check_params.check_accuracy_of_label(numbertestcase,1,test_result))
print(check_params.check_accuracy_of_label(numbertestcase,2,test_result))

print("-------------------------------------------------------------------------------------------------------------")


model_name = '../models_bin/RandomForest_with_tfidf.joblib'
dump(RandomForestModel,model_name)

    

# link_exp = "experiments/rf_tf_exp_start%s_length_%s.txt"%(str(train_data_begin_index),(str((train_data_end_index)*3)))
# f = open(link_exp,"w+",encoding="utf-8")
# max_leaf_nodes=[8,10,12]
# max_depth=[6,7,8]
# n_estimators=[35,40,45]
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


