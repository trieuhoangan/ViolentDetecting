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


def create_model(max_words, embedding_dim, maxlen, num_class=3):
	#most simple model

	model = Sequential()
	model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
	model.add(Flatten())
	model.add(Dense(512, activation='relu', input_shape=(max_words,)))
	model.add(Dropout(0.5))
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_class, activation='softmax'))

	opt = RMSprop(lr=0.0001)
	model.compile(
		loss='sparse_categorical_crossentropy',
		optimizer=opt,
		metrics=["accuracy"]
	)
	print(model.summary())

	return model

def create_lstm_model(max_words, embedding_dim, maxlen, num_class=3):

	model = Sequential()
	model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
	# model.add(SpatialDropout1D(0.2))
	model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_class, activation='softmax'))
	opt = RMSprop(lr=0.0001)
	model.compile(
		loss='sparse_categorical_crossentropy',
		optimizer=opt,
		metrics=["accuracy"]
	)
	print(model.summary())

	return model

def plot_train_val(history, value="loss"):
	"""Print loss or accuracy
	"loss" or "acc"
	"""
	train_value = history.history[value]
	val_value = history.history["val_"+value]

	epochs = range(1, len(train_value) + 1)
	# "bo" is for "blue dot"
	plt.plot(epochs, train_value, 'r+', label='Training %s' % value)
	# b is for "solid blue line"
	plt.plot(epochs, val_value, 'b', label='Validation %s' % value)
	plt.title('Training and validation %s' % value)
	plt.xlabel('Epochs')
	plt.ylabel(value)
	plt.legend()

	plt.show()


#load data
datapickle_file = "violence_data.pkl"
print("Loading data to file %s" % datapickle_file)
(x_train, y_train, x_test, y_test) = cloudpickle.load(open(datapickle_file, 'rb'))


max_words = 10000
maxlen = 200
embedding_dim = 50
model = create_lstm_model(max_words, embedding_dim, maxlen, num_class=3)

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=7)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)
history = model.fit(x_train, y_train,
                    epochs=50,
                    batch_size=128,
                    validation_split=0.2,
                    verbose=2,
                    callbacks=[reduce_lr, es, mc])
print(model.evaluate(x_test, y_test, verbose=1))

plot_train_val(history, "loss")
