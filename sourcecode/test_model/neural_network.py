from __future__ import print_function

import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D,RNN
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.initializers import Constant
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.preprocessing import OneHotEncoder
from gensim.models import KeyedVectors
sys.path.append("..")
from customlib import vectorizeText,accessMysql,check_params
print('vectorize sentences')
model_link = '../models_bin/wiki.vi.model.bin'
word2vec_model = KeyedVectors.load_word2vec_format(model_link, binary=True)
MAX_SEQUENCE_LENGTH=400
data = accessMysql.getContentList("select * from sentences")
label_list = accessMysql.getLabelList("select * from sentences")

datas = []
for text in data:
    text = text.replace('_',' ')
    datas.append(text)
tokens = vectorizeText.split_list(datas)
vectors = [] 
for token in tokens:
    vectors.append(vectorizeText.sent_vectorize(token,word2vec_model))
vectors = np.nan_to_num(vectors)
labels=[]
for label in label_list:
    one_hot_vector = np.zeros(3, dtype=int)
    one_hot_vector[label]=1
    labels.append(one_hot_vector)
labels= np.array(labels)

vectors = np.array(vectors)
texts,test_data,train_labels,test_labels = train_test_split(vectors,labels,test_size=0.1,random_state=20)
x_train,x_val,y_train,y_val = train_test_split(texts,train_labels,test_size=0.1,random_state=20)
# print(y_val)


embedding_layer = word2vec_model.wv.get_keras_embedding(train_embeddings=False)

print('Training model.')

# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')
x = GlobalMaxPooling1D()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(3, activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

model.fit(x_train, y_train,
          batch_size=128,
          epochs=7,
          validation_data=(x_val, y_val))
test_result = model.predict(test_data)
# accurracy = "accuracy score : "+ str(accuracy_score(test_labels,test_result))
print(test_result)
