import re
import sys
sys.path.append("..")

from sklearn.model_selection import GridSearchCV
from customlib import vectorizeText,accessMysql,check_params
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
import numpy as np

from joblib import load
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

print('training model ')
test_vectors = tf.transform(test_data)
scores = ['precision', 'recall']
params = [{'random_state':[8]}]
model = DecisionTreeClassifier(random_state=8).fit(vectors, labels)


# for score in scores:
#     model = GridSearchCV(DecisionTreeClassifier(),params,cv=5,scoring='%s_macro' % score).fit(vectors, labels)

print('testing')

test_result = model.predict(test_vectors)
print(accuracy_score(test_labels,test_result))
print("test result:")
# print(test_result)
print("--------------------------------------------------------------")
print("test ex amples:")
# print(test_labels)
# print(model.best_params_)
accurracy = "accuracy score : "+ str(accuracy_score(test_labels,test_result))
print(accurracy)
print(check_params.check_accuracy_of_label(numbertestcase,0,test_result))
print(check_params.check_accuracy_of_label(numbertestcase,1,test_result))
print(check_params.check_accuracy_of_label(numbertestcase,2,test_result))
print("-------------------------------------------------------------------------------------------------------------")
    
from joblib import dump
model_name = '../models_bin/RandomForest_with_tfidf.joblib'
dump(model,model_name)
