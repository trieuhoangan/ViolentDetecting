import pymysql.cursors
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict
from gensim import models
from gensim.models import KeyedVectors

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

def get_vector(sen,model):
    sen = sen.replace('_',' ')
    words = sen.split(' ')
    sen_vector = np.zeros(400)
    
    word_vector = model.wv
    for word in words:
        if word !=' ':
            if word in word_vector:
                tmp_vector = np.zeros(400)
                tmp_vector = np.add(tmp_vector,model[word])
                tmp_vector = tmp_vector/np.sqrt(tmp_vector.dot(tmp_vector))
                sen_vector = np.add(sen_vector,tmp_vector)
    return sen_vector

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

sentence_label_1 = accessMysql.getContentListWithLimit("select * from sentences where label = 1 and istrain = 1  limit %s,%s",train_data_begin_index,train_data_end_index)
label_type_1 = accessMysql.getLabelListWithLimit("select * from sentences where label = 1 and istrain = 1  limit %s,%s",train_data_begin_index,train_data_end_index)

test_data_label_type_1 = accessMysql.getContentListWithLimit("select * from sentences where label = 1 and istrain = 1  limit %s,%s",test_data_begin_index,test_data_end_index)
test_label_type_1 = accessMysql.getLabelListWithLimit("select * from sentences where label = 1 and istrain = 1  limit %s,%s",test_data_begin_index,test_data_end_index)

sentence_label_2 = accessMysql.getContentListWithLimit("select * from sentences where label = 2 and istrain = 1  limit %s,%s",train_data_begin_index,train_data_end_index)
label_type_2 = accessMysql.getLabelListWithLimit("select * from sentences where label = 2 and istrain = 1  limit %s,%s",train_data_begin_index,train_data_end_index)
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
import re
clean_spe_char = re.compile('[\\~#%&*{}/:<>?,.;|"]')
for text in texts:
    text = re.sub(clean_spe_char,' ',text)
    # text = text.replace("'",' ')
    text = text.replace('_',' ')
    datas.append(text)
test_cases = []
for test_case in test_data:
    test_case =  re.sub(clean_spe_char,' ',test_case)
    # test_case = test_case.replace("'",' ')
    test_cases.append(test_case.replace('_',' '))


print('vectorize sentences')
model_link = '../models_bin/wiki.vi.model.bin'
word2vec_model = KeyedVectors.load_word2vec_format(model_link, binary=True)

vectors = [] 
for token in datas:
    vectors.append(get_vector(token,word2vec_model))
# vectors = np.nan_to_num(vectors)


test_vectors = []
for test_sequence in test_cases:
    test_vectors.append(get_vector(test_sequence,word2vec_model))
# test_vectors = np.nan_to_num(test_vectors)
# print(vectors.shape)
print('train model')
scores = ['precision', 'recall']
params = [{'penalty':['l1'],'solver':['saga'],'multi_class':['multinomial','ovr']},{'penalty':['l2'],'solver':['lbfgs','newton-cg','sag'],'random_state':[0,2,4],'max_iter':[280,300,320],'multi_class':['multinomial','ovr']},{'penalty':['l1'],'solver':['liblinear'],'multi_class':['ovr']}]


# for score in scores:

# model = GridSearchCV(LogisticRegression(),params,cv=5,scoring='%s_macro' % score).fit(vectors,labels)
model = LogisticRegression(penalty='l2',solver='lbfgs',max_iter=2000,multi_class='ovr',random_state=0)

print(cross_val_score(model,vectors,labels,cv=5))
model.fit(vectors,labels)
print("testing")
test_result = model.predict(test_vectors)
# test_result = cross_val_predict(model,vectors,labels,cv=5)
# print("test result:")

# print("--------------------------------------------------------------")
# print("test examples:")


accurracy = "accuracy score : "+ str(accuracy_score(test_labels,test_result))
f1_score_macro = "f1 score ave macro : "+str(f1_score(test_labels,test_result,average=None))

# print(model.best_params_)
print("--------------------------------------------------------------")
print(accurracy)
print(f1_score_macro)

print(check_params.check_accuracy_of_label(numbertestcase,0,test_result))
print(check_params.check_accuracy_of_label(numbertestcase,1,test_result))
print(check_params.check_accuracy_of_label(numbertestcase,2,test_result))
    
from joblib import dump
model_name = '../models_bin/losgisticRegression_with_word2vec.joblib'
dump(model,model_name)

