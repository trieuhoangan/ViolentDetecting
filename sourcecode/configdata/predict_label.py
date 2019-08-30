import pickle
import pymysql.cursors
import sys
import numpy as np
from joblib import dump, load
sys.path.append("..")
from customlib import vectorizeText,accessMysql
from gensim import models
from gensim.models import KeyedVectors

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
print('loading models')
LogisticModel_tfidf_link = '../models_bin/losgisticRegression_with_tfidf.joblib'
LogisticModel_word2vec_link = '../models_bin/losgisticRegression_with_word2vec.joblib'
RandomForest_tdidf_link = '../models_bin/RandomForest_with_tfidf.joblib'
RandomForest_word2vec_link = '../models_bin/RandomForest_with_word2vec.joblib'
Gaussian_tfidf_link = '../models_bin/gaussian_with_tfidf.joblib'
Gaussian_word2vec_link = '../models_bin/gaussian_with_word2vec.joblib'
word2vec_model_link = '../models_bin/wiki.vi.model.bin'
tfidf_vectorize_model_link = '../models_bin/tfidf_model.joblib'
word2vec_model = KeyedVectors.load_word2vec_format(word2vec_model_link, binary=True)

Tfidf_model_links = [LogisticModel_tfidf_link,RandomForest_tdidf_link]
word2vec_model_link = [LogisticModel_word2vec_link,RandomForest_word2vec_link]

tfidf_vectorize_model = load(tfidf_vectorize_model_link)
word2vec_models = []
Tfidf_models=[]
# test_sen = ['mèo béo ngốc nghếch']
# words = []
# for sen in test_sen:
#     words.append(sen.split(' '))
# test_vec = []
# for word in words : 
#     test_vec.append(vectorizeText.sent_vectorize(word,word2vec_model))
for link in word2vec_model_link:
    with open(link,'rb') as f:
        print(link)
        model = load(link)
        # print(model.predict(test_vec))
        word2vec_models.append(model)
# testvector = tfidf_vectorize_model.transform(test_sen)
for link in Tfidf_model_links:
    with open(link,'rb') as f:
        print(link)
        model = load(link)
        # print(model.predict(testvector))
        Tfidf_models.append(model)
# Gaussian_tfidf_model = load(Gaussian_tfidf_link)
print(Gaussian_tfidf_link)
# print(Gaussian_tfidf_model.predict(testvector.toarray()))
print('loading database')
connection = pymysql.connect(host='localhost',
                                user='root',
                                password='12345678',
                                db='AliceII',
                                charset='utf8',
                                cursorclass=pymysql.cursors.DictCursor)
try:
    with connection.cursor() as cursor:
        # print('loading database 2')
        counter = 0
        sql='select * from sentences where label is null or label > 2'
        cursor.execute(sql)
        results = cursor.fetchall()
        # print(len(results))
        for result in results:
            content = str(result.get('content'))
            sen = [content]
            tfidf_vector = tfidf_vectorize_model.transform(sen)
            words = [content.split(' ')]
            word2vec_vector=[]
            for word in words:
                word2vec_vector.append(vectorizeText.sent_vectorize(word,word2vec_model))
            word2vec_vector = np.nan_to_num(word2vec_vector)    
            # gaussian_array = tfidf_vector.toarray()
            labels = []

            print('predicting')
            for model in word2vec_models:
                labels.append(model.predict(word2vec_vector))
            for model in Tfidf_models:
                labels.append(model.predict(tfidf_vector))
            # labels.append(Gaussian_tfidf_model.predict(gaussian_array))
            # print(content)
            LogisticModel_word2vec_label = str(labels[0][0])
            RandomForest_word2vec_label = str(labels[1][0])
            # Gaussian_word2vec_label = str(labels[2][0])
            LogisticModel_tfidf_label = str(labels[2][0])
            RandomForest_tdidf_label = str(labels[3][0])
            # Gaussian_tfidf_label = str(labels[5][0])
            Gaussian_tfidf_label = -1
            Gaussian_word2vec_label = -1
            average_label = 0
            count_label_1 = 0
            for label in labels:
                # print(str(label[0]))
                if(label[0] == 1):
                    count_label_1 = count_label_1+1
                average_label = average_label+label[0]
            average_label = average_label/4
            # print(average_label)
            _id = result.get('id')

            # if(average_label == 0):
            if(count_label_1 ==4): 
                sql = 'update sentences set label = 1 where id = %s'
                cursor.execute(sql,(_id))
                counter = counter + 1
            sql = 'update sentences set RandomForest_tfidf_label = %s,RandomForest_word2vec_label=%s,LogisticRegression_tfidf_label = %s,LogisticRegression_word2vec_label = %s,Gaussian_word2vec_label=%s,Gaussian_tfidf_label=%s,average_label = %s where id = %s'
            print(".")
            cursor.execute(sql,(RandomForest_tdidf_label,RandomForest_word2vec_label,LogisticModel_tfidf_label,LogisticModel_word2vec_label,Gaussian_word2vec_label,Gaussian_tfidf_label,str(average_label),_id))
    print(counter)
    connection.commit()
finally:
    connection.close()