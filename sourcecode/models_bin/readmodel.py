import numpy as np
import pymysql.cursors
embeddings_index = {}
with open('wiki.vi.model.bin','rb') as f:
    i =0
    for line in f:
        values = line.split()
        # word = values[0]
        # coefs = np.asarray(values[1:], dtype='char64')
        embeddings_index[i] = line
        i=i+1
        # print(line)

from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format('wiki.vi.model.bin')
texts = []
labels = []
connection = pymysql.connect(host='localhost',
                            user='root',
                            password='12345678',
                            db='AliceII',
                            charset='utf8',
                            cursorclass=pymysql.cursors.DictCursor)
                
try:
    with connection.cursor() as cursor:
        sql = "select * from newspaper where label is not null and id = 2"
        cursor.execute(sql)
        results = cursor.fetchall()
        for result in results:
            if(int(result.get('label'))>0):
                texts.append(result.get('content'))
                label_id = result.get('label')
                labels.append(int(label_id)-1)

    connection.commit()
finally:
    connection.close()

