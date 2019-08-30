import numpy as np
from joblib import dump,load
from sklearn.ensemble import VotingClassifier
from gensim.models import KeyedVectors
from sklearn.model_selection import GridSearchCV
import sys
from sklearn.metrics import f1_score
sys.path.append("..")
from customlib import vectorizeText,accessMysql,check_params
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from argparse import ArgumentParser
import re
import subprocess

tokenizer_path = r'D:\SkyLabDocument\JVnTextPro\JVnTextPro-3.0.3-executable.jar'
# Classifier models
LogisticModel_tfidf_link = './models_bin/losgisticRegression_with_tfidf.joblib'
LogisticModel_word2vec_link = './models_bin/losgisticRegression_with_word2vec.joblib'
RandomForest_tdidf_link = './models_bin/RandomForest_with_tfidf.joblib'
RandomForest_word2vec_link = './models_bin/RandomForest_with_word2vec.joblib'
Gaussian_tfidf_link = './models_bin/gaussian_with_tfidf.joblib'
Gaussian_word2vec_link = './models_bin/gaussian_with_word2vec.joblib'
SVC_tfidf_link = './models_bin/SVC_with_tfidf.joblib'
SVC_word2vec_link = './models_bin/SVC_with_word2vec.joblib'
ensemble_hard_voting_w2v_link = './models_bin/ensemble_hardvoting_with_word2vec.joblib'
ensemble_hard_voting_tf_link  = './models_bin/ensemble_hardvoting_with_tf.joblib'
ensemble_soft_voting_w2v_link = './models_bin/ensemble_softvoting_with_word2vec.joblib'
ensemble_soft_voting_tf_link = './models_bin/ensemble_softvoting_with_tf.joblib'

# Vectorize models
word2vec_vectorize_model_link = './models_bin/wiki.vi.model.bin'
tfidf_vectorize_model_link = './models_bin/tfidf_model.joblib'


#load Vectorize models
tf = load(tfidf_vectorize_model_link)
word2vec = KeyedVectors.load_word2vec_format(word2vec_vectorize_model_link, binary=True)

#load classifier model
word2vec_model_links = [LogisticModel_word2vec_link,SVC_word2vec_link,RandomForest_word2vec_link,ensemble_hard_voting_w2v_link,ensemble_soft_voting_w2v_link]
tfidf_model_links = [LogisticModel_tfidf_link,SVC_tfidf_link,RandomForest_tdidf_link,ensemble_hard_voting_tf_link,ensemble_soft_voting_tf_link]
word2vec_models = []
Tfidf_models=[]

for link in word2vec_model_links:
    with open(link,'rb') as f:
        # print(link)
        model = load(link)
        # print(model.predict(test_vec))
        word2vec_models.append(model)
for link in tfidf_model_links:
    with open(link,'rb') as f:
        # print(link)
        model = load(link)
        # print(model.predict(testvector))
        Tfidf_models.append(model)



def read(filename):
    subprocess.call(['java', '-jar', tokenizer_path, '-senseg','-wordseg','-input',filename])
    newFileName = filename+".pro"
    w2v_vectors = []
    tf_vectors = []
    sentences = []

    with open(newFileName,'r',encoding="utf-8",errors ='replace') as f:
        sentences = f.readlines()
        
        tf_vectors = tf.transform(sentences)
        for sentence in sentences:
            # print(sentence)
            valid_file_name_character = re.compile('[\\~#%&*{}/:<>?|\"-]')
            sentence = re.sub(valid_file_name_character,'',sentence)
            sentence=sentence.replace('\n',' ')
            # print(sentence)
            w2v_vectors.append(vectorizeText.sent_vectorize(sentence.split(' '),word2vec))
        w2v_vectors = np.nan_to_num(w2v_vectors)
        # print("tf vector: \n")
        # print(tf_vectors.shape)
        # print('\n')
        # print("w2v vector: \n")
        # print()
    results=[]
    for model in word2vec_models:
        results.append(model.predict(w2v_vectors))
    for model in Tfidf_models:
        results.append(model.predict(tf_vectors))

    with open("result_"+filename,'w',encoding="utf-8",errors ='replace') as f:
        f.write("câu,LogisticModel_word2vec,SVC_word2vec,RandomForest_word2vec,ensemble_hard_voting_w2v,ensemble_soft_voting_w2v,LogisticModel_tfidf,SVC_tfidf,RandomForest_tdidf,ensemble_hard_voting_tfidf,ensemble_soft_voting_tfidf\n")
        i = 0
        for sentence in sentences:
            sentence = sentence.replace('\n','')
            sentence = '"'+sentence+'"'
            for result in results:
                sentence = sentence+',' +str(result[i])
            i=i+1
            f.write(sentence+"\n")
    # print(Tfidf_models[4].predict(tf_vectors))
    # print(result)
        

if __name__ == "__main__":
    # parser = ArgumentParser()
    # parser.add_argument("-f", "--file", dest="filename",
    #                     help="write report to FILE", metavar="FILE")
    # parser.add_argument("-q", "--quiet",
    #                     action="store_false", dest="verbose", default=True,
    #                     help="don't print status messages to stdout")
    # args = parser.parse_args()
    # with open()
    read(sys.argv[1])