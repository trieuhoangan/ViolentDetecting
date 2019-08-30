
import codecs
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
from distutils.version import LooseVersion, StrictVersion
import numpy

model = 'wiki.vi.model.bin'
from gensim.models import KeyedVectors
word2vec_model = KeyedVectors.load_word2vec_format(model, binary=True)
# print('input the word')
# word = input()
# output = word2vec_model.most_similar(word,topn =20)
print(len(word2vec_model['ăn']))
