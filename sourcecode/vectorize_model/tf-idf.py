import sys
sys.path.append("..")
from sklearn.feature_extraction.text import TfidfVectorizer
from customlib import vectorizeText,accessMysql
from joblib import dump,load
import re
all_text = accessMysql.getContentList("select * from sentences")
valid_file_name_character = re.compile('[\\~#%&*{}/:<>?|\"-]')
words = []
for text in all_text:
    text = re.sub(valid_file_name_character,'',text)
    words.append(text.split(' '))
    # vowels = text.split(' ')
    # for vowel in vowels:
    #     vowel = vowel.replace("_"," ")
    # words.append(vowels)
stop_word = ['.',',',';','!','@','#','-','>','(',')','/']
tf = TfidfVectorizer(min_df=400,max_df=0.15,sublinear_tf=True,encoding='utf-8',stop_words=stop_word,analyzer='word').fit(all_text)

model_link = '../models_bin/tfidf_model.joblib'
dump(tf,model_link)


