
###############################################################################
# Training Word2Vec to get word embedding vector
#
# Some code are taken from:
# https://github.com/wendykan/DeepLearningMovies 
###############################################################################
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import re
import logging

import pymysql.cursors
from gensim.models import Word2Vec
from bs4 import BeautifulSoup


def strip_tags(html):
    "Strip html tags"
    return BeautifulSoup(html,"lxml").get_text(' ')

def text_to_token(text):
    "Get list of token for training"
    # Strip HTML
    text = strip_tags(text)
    # Keep only word
    text = re.sub("\W", " ", text)
    # Lower and split sentence
    token = text.lower().split()
    # Don't remember the number
    for i in range(len(token)):
        token[i] = len(token[i])*'DIGIT' if token[i].isdigit() else token[i]
    return token

def read_sentences():
    fp='data/data.txt'
    # "Read and split token from text file"
    sentences = []
    with open(fp, 'r',encoding="utf8") as f:
        for line in f:
            if '|' not in line: # Remove menu items in some newspaper
                sentences.append(text_to_token(line.strip()))
    return sentences

if __name__ == '__main__':
    # Read data from files
    sentences = read_sentences()
    print('Loaded {} sentences!'.format(len(sentences)))
    # Import the built-in logging module and configure it so that Word2Vec
    # creates nice output messages
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
        level=logging.INFO)
    # Set values for various parameters
    num_features = 100    # Word vector dimensionality
    min_word_count = 7    # Minimum word count
    num_workers = 4       # Number of threads to run in parallel
    context = 10          # Context window size
    downsampling = 1e-3   # Downsample setting for frequent words
    print("Training Word2Vec model...")
    # Initialize and train the model
    model = Word2Vec(sentences, workers=num_workers, \
                size=num_features, min_count = min_word_count, \
                window = context, sample = downsampling, seed=1)
    model.init_sims(replace=True)
    model_name = "data/{}features_{}context_{}minwords.vec".format(
        num_features, context, min_word_count
    )
    model.save(model_name)




