from gensim import models
from gensim.models import KeyedVectors
from joblib import load
import numpy as np 
import sys
sys.path.append("..")
from customlib import vectorizeText,accessMysql,check_params

tfidf_vectorize_model_link = '../models_bin/tfidf_model.joblib'
tf = load(tfidf_vectorize_model_link)

model_link = '../models_bin/wiki.vi.model.bin'
word2vec_model = KeyedVectors.load_word2vec_format(model_link, binary=True)
word_vector = word2vec_model.wv 
sens = []
with open("test_doc_1.txt",'r',encoding="utf-8") as f:
    sens = f.readlines()
with open("test_vector_1.txt",'w',encoding="utf-8") as f:
    for sen in sens:
        sen = sen.replace('\n','')
        words = sen.split(' ')
        for word in words:
            if word in word_vector:
                word_vec = word2vec_model[word.lower()]
                word_vec = word_vec/np.sqrt(word_vec.dot(word_vec))
                f.write(word)
                f.write(str(word_vec))
                f.write('\n')
        # sent_vec = np.zeros(400)
        # try:
        #     sent_vec = np.add(sent_vec,word2vec_model[sen])
        # except:
        #     pass
        # sent_vec =  sent_vec/np.sqrt(sent_vec.dot(sent_vec))
        
        f.write('\n\n\n')
    f.write("\n\n\n\n")
    for sen in sens:
        f.write(str(tf.transform(sen.split(" "))))
        f.write('\n\n')
    