import pymysql.cursors
import nltk
# nltk.download()
from nltk.tokenize import sent_tokenize

if __name__ == '__main__':
    connection = pymysql.connect(host='localhost',
                                user='root',
                                password='12345678',
                                db='AliceII',
                                charset='utf8',
                                cursorclass=pymysql.cursors.DictCursor)
    try:
        with connection.cursor() as cursor:
            i = 1
            while i<6000:
                sql = "select * from newspaper where label is not null"
                cursor.execute(sql,(i))
                result = cursor.fetchall()
                content = result.get('content')
                content = content.replace('\n','')
                filename = "data/data.txt"
                with open(filename,'a+',encoding="utf8") as appenddata:
                    sentences = sent_tokenize(content)
                    for sentence in sentences:
                        appenddata.write(sentence+'\n')
                i=i+1
                print("loading....%s\n"%i)
        connection.commit()
    finally:
        connection.close()