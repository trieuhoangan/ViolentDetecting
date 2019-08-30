import os
import numpy as np
import pandas as pd
import string
import random

import matplotlib.pyplot as plt
import cloudpickle


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, Dropout
from keras.layers import LSTM, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras import regularizers
from keras.optimizers import RMSprop, Adam
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model




STOPWORDS = string.punctuation + '“”'
train_ratio = 0.8

def clean_sent(s):
    s = s.strip()
    s = s.split(" ")
    s = [i for i in s if not i in STOPWORDS]
    return " ".join(s)

def load_csv(fname, shuffle=True):
    df = pd.read_csv(fname)
    # print(df.columns)
    # ids = df.newspaper_id.tolist()
    sents = df.content.tolist()
    sents = [clean_sent(s) for s in sents]
    #group by ids
    # values = set(ids)
    # sent_groups = [[s for i,s in enumerate(sents) if ids[i] == x] for x in values]
    random.shuffle(sents)

    return sents

def average_length(sents):
    length = [len(s) for s in sents]
    return np.mean(length)


def split_train_test(sents, label_id):
    split_id = round(len(sents)*train_ratio)
    x_train = sents[:split_id]
    x_test = sents[split_id:]
    y_train = [label_id] *len(x_train)
    y_test = [label_id] *len(x_test)
    return x_train, y_train, x_test, y_test

def run():
    x_train =[]
    y_train =[]
    x_test = []
    y_test = []

    labels = [0, 2]
    for label in labels:
        fname = "violence-data/label_{}.csv".format(label)
        sents = load_csv(fname)
        print("Label: {}, len: {}".format(label, len(sents)))
        trainx, trainy, testx, testy = split_train_test(sents, label)
        x_train = x_train + trainx
        y_train = y_train + trainy
        x_test = x_test + testx
        y_test = y_test + testy


    print(len(x_train), average_length(x_train))
    print(len(x_test), average_length(x_test))


    #para
    max_words = 10000
    maxlen = 200
    embedding_dim = 50
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(x_train)
    train_sequences = tokenizer.texts_to_sequences(x_train)
    x_train = pad_sequences(train_sequences, maxlen=maxlen)

    test_sequences = tokenizer.texts_to_sequences(x_test)
    x_test = pad_sequences(test_sequences, maxlen=maxlen)

    #tfidf
    # x_train = tokenizer.texts_to_matrix(x_train, mode='tfidf')
    # x_test = tokenizer.texts_to_matrix(x_test, mode='tfidf')
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train).reshape(len(y_train), 1)
    y_test = np.array(y_test).reshape(len(y_test), 1)

    # le = LabelEncoder()
    # y_train = le.fit_transform(y_train)
    # y_test = le.transform(y_test)

    print(x_train.shape)
    print(y_train.shape)

    datapickle_file = "violence_data.pkl"
    dataset = (x_train, y_train, x_test, y_test)
    cloudpickle.dump(dataset, open(datapickle_file, 'wb'))
    print("Dump data to file %s" % datapickle_file)

if __name__ == '__main__':
    run()
