from gensim import models
from gensim.models import KeyedVectors
from joblib import load

tfidf_vectorize_model_link = '../models_bin/tfidf_model.joblib'
tf = load(tfidf_vectorize_model_link)

model_link = '../models_bin/wiki.vi.model.bin'
word2vec_model = KeyedVectors.load_word2vec_format(model_link, binary=True)

import sys
sys.path.append("..")
from customlib import vectorizeText,accessMysql,check_params

sens = accessMysql.getContentList("select * from aliceii.sentences where newspaper_id = 2796")

with open("test_doc_2.txt",'w',encoding="utf-8") as f:
    for sen in sens:
        f.write(sen.replace("_"," "))