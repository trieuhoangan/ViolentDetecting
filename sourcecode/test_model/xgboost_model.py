import pymysql.cursors
import numpy as np
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict
from gensim import models
from gensim.models import KeyedVectors
import xgboost as xg
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
# sentence_label_0 = accessMysql.getContentListWithLimit("select * from sentences where label = 0 and istrain = 1 limit %s,%s",train_data_begin_index,train_data_end_index)
# label_type_0 = accessMysql.getLabelListWithLimit("select * from sentences where label = 0 and istrain = 1  limit %s,%s",train_data_begin_index,train_data_end_index)
# test_data_label_type_0 = accessMysql.getContentListWithLimit("select * from sentences where label = 0 and istrain = 1  limit %s,%s",test_data_begin_index,test_data_end_index)
# test_label_type_0 = accessMysql.getLabelListWithLimit("select * from sentences where label = 0 and istrain = 1  limit %s,%s",test_data_begin_index,test_data_end_index)

# sentence_label_1 = accessMysql.getContentListWithLimit("select * from sentences where label = 1 and istrain = 1  limit %s,%s",train_data_begin_index,train_data_end_index)
# # sentence_label_1_part2 = accessMysql.getContentListWithLimit("select * from sentences where label = 1 limit %s,%s",700,1100)
# label_type_1 = accessMysql.getLabelListWithLimit("select * from sentences where label = 1 and istrain = 1  limit %s,%s",train_data_begin_index,train_data_end_index)
# # label_type_1_part2 = accessMysql.getLabelListWithLimit("select * from sentences where label = 1  limit %s,%s",700,1100)
# test_data_label_type_1 = accessMysql.getContentListWithLimit("select * from sentences where label = 1 and istrain = 1  limit %s,%s",test_data_begin_index,test_data_end_index)
# test_label_type_1 = accessMysql.getLabelListWithLimit("select * from sentences where label = 1 and istrain = 1  limit %s,%s",test_data_begin_index,test_data_end_index)

# sentence_label_2 = accessMysql.getContentListWithLimit("select * from sentences where label = 2 and istrain = 1  limit %s,%s",train_data_begin_index,train_data_end_index)
# # sentence_label_2_part2 = accessMysql.getContentListWithLimit("select * from sentences where label = 2 limit %s,%s",700,1100)
# label_type_2 = accessMysql.getLabelListWithLimit("select * from sentences where label = 2 and istrain = 1  limit %s,%s",train_data_begin_index,train_data_end_index)
# # label_type_2_part2 = accessMysql.getLabelListWithLimit("select * from sentences where label = 2  limit %s,%s",700,1100)
# test_data_label_type_2 = accessMysql.getContentListWithLimit("select * from sentences where label = 2 and istrain = 1  limit %s,%s",test_data_begin_index,test_data_end_index)
# test_label_type_2 = accessMysql.getLabelListWithLimit("select * from sentences where label = 2 and istrain = 1  limit %s,%s",test_data_begin_index,test_data_end_index)
# texts = []
# labels = []
# test_data = []
# test_labels = []
# # print(label_type_0)
# texts.extend(sentence_label_0)
# texts.extend(sentence_label_1)
# texts.extend(sentence_label_2)
# # texts.extend(sentence_label_1_part2)
# # texts.extend(sentence_label_2_part2)
# labels.extend(label_type_0)
# labels.extend(label_type_1)
# labels.extend(label_type_2)
# # labels.extend(label_type_1_part2)
# # labels.extend(label_type_2_part2)
# test_data.extend(test_data_label_type_0)
# test_data.extend(test_data_label_type_1)
# test_data.extend(test_data_label_type_2)
# test_labels.extend(test_label_type_0)
# test_labels.extend(test_label_type_1)
# test_labels.extend(test_label_type_2)

content = accessMysql.getContentList("select * from sentences")
label_of_content = accessMysql.getLabelList("select * from sentences")
# test_data = accessMysql.getContentList("select * from sentences where istrain = 2 ")
# test_labels = accessMysql.getLabelList("select * from sentences where istrain = 2 ")
texts,test_data,labels,test_labels = train_test_split(content,label_of_content,test_size=0.12,random_state=42)
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
test_sequence_list = vectorizeText.split_list(test_cases)
# print(len(test_data))
print('vectorize sentences')
model_link = '../models_bin/wiki.vi.model.bin'
word2vec_model = KeyedVectors.load_word2vec_format(model_link, binary=True)

vectors = [] 
for token in tokens:
    vectors.append(vectorizeText.sent_vectorize(token,word2vec_model))
vectors = np.nan_to_num(vectors)
test_vectors = []
for test_sequence in test_sequence_list:
    test_vectors.append(vectorizeText.sent_vectorize(test_sequence,word2vec_model))
test_vectors = np.nan_to_num(test_vectors)
print(vectors.shape)
# print(test_vectors)
print('train model')
scores = ['precision', 'recall']
params = [{'max_depth':[1,2,3],'n_estimators ':[30,40,50]}]

# print(test_vectors)
# for score in scores:

# model = GridSearchCV(xg.XGBRFRegressor(),params,cv=5).fit(vectors,labels)
# model = LogisticRegression(penalty='l2',solver='lbfgs',max_iter=280,multi_class='ovr',random_state=0)

model = xg.XGBRFClassifier()
# print(cross_val_score(model,vectors,labels,cv=5))
model.fit(vectors,labels)
test_result = model.predict(test_vectors)

# test_result = cross_val_predict(model,vectors,labels,cv=5)
# print("test result:")

# print("--------------------------------------------------------------")
# print("test examples:")


accurracy = "accuracy score : "+ str(accuracy_score(test_labels,test_result))
f1_score_macro = "f1 score ave macro : "+str(f1_score(test_labels,test_result,average=None))
# f1_score_micro = "f1 score ave micro : "+str(f1_score(test_labels,test_result,average='micro'))
# f1_score_weighted = "f1 score ave weighted : "+str(f1_score(test_labels,test_result,average='weighted'))
# precision_score_macro = "precision score ave macro : " + str(precision_score(test_labels,test_result,average='macro'))
# precision_score_micro = "precision score ave micro : " + str(precision_score(test_labels,test_result,average='micro'))
# precision_score_weighted = "precision score ave weighted : " + str(precision_score(test_labels,test_result,average='weighted'))
# print(model.best_params_)
print("--------------------------------------------------------------")
print(accurracy)
print(f1_score_macro)
# print(f1_score_micro)
# print(f1_score_weighted)
# print(precision_score_macro)
# print(precision_score_micro)
# print(precision_score_weighted)
# print(check_params.check_accuracy_of_label(numbertestcase,0,test_result))
# print(check_params.check_accuracy_of_label(numbertestcase,1,test_result))
# print(check_params.check_accuracy_of_label(numbertestcase,2,test_result))
    
from joblib import dump
model_name = '../models_bin/xgboost.joblib'
dump(model,model_name)

####### For experiments only 
# link_exp = "experiments/lr_w_exp_start%s_length_%s.txt"%(str(train_data_begin_index),(str((train_data_end_index)*3)))
# f = open(link_exp,"w+",encoding="utf-8")
# penalty=['l1','l2']
# solver=['liblinear','saga','lbfgs','newton-cg','sag']
# multi_class=['multinomial','ovr']
# random_state=[0,2,4]
# max_iter=[280,300,320]
# model = LogisticRegression()
# for pel in penalty:
#     for iters in max_iter:
#         for state in random_state:
#             for classes in multi_class:
#                 for sol in solver:
#                     if pel=='l1':
#                         if sol=='saga':
#                             model.set_params(penalty=pel,solver=sol,multi_class=classes)
#                         else :
#                             if sol=="libliner":
#                                 model.set_params(penalty=pel,solver=sol,multi_class="ovr")
#                             else :
#                                 continue
#                     else :
#                         if sol == "lbfgs" or sol == "newton-cg" or sol =='sag':
#                             model = model.set_params(penalty=pel,solver=sol,multi_class=classes,max_iter=iters,random_state=state)
#                         else:
#                             continue
#                     model.fit(vectors,labels)
#                     test_result = model.predict(test_vectors)
#                     accurracy = "accuracy score : "+ str(accuracy_score(test_labels,test_result))
#                     f1_score_macro = "f1 score ave macro : "+str(f1_score(test_labels,test_result,average='macro'))
#                     f1_score_micro = "f1 score ave micro : "+str(f1_score(test_labels,test_result,average='micro'))
#                     f1_score_weighted = "f1 score ave weighted : "+str(f1_score(test_labels,test_result,average='weighted'))
#                     precision_score_macro = "precision score ave macro : " + str(precision_score(test_labels,test_result,average='macro'))
#                     precision_score_micro = "precision score ave micro : " + str(precision_score(test_labels,test_result,average='micro'))
#                     precision_score_weighted = "precision score ave weighted : " + str(precision_score(test_labels,test_result,average='weighted'))
#                     params = "pel : "+pel+" , multi_class : "+classes+" , solver : "+sol+" , max_iter : "+str(iters)+" , random_state : "+str(state)+" , "+accurracy+" , "+f1_score_macro+"\n"
#                     f.write(params)


# f.close