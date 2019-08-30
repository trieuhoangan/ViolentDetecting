import pymysql.cursors
import sys
sys.path.append("..")
from customlib import accessMysql
dbname = 'AliceII'
connection = pymysql.connect(host='localhost',
                            user='root',
                            password='12345678',
                            db= dbname,
                            charset='utf8',
                            cursorclass=pymysql.cursors.DictCursor)
try:
    with connection.cursor() as cursor:
            print('loading data')
            sql = "select * from newspaper where id not in ( select distinct newspaper_id from sentences)"
            cursor.execute(sql)
            results = cursor.fetchall()
            print("start to extract")
            for result in results:
                content = str(result.get('content'))
                content = accessMysql.normalizeContent(content)
                sentences = content.split('.')
                # print(sentences)
                newspaper_id = result.get("id")
                for sentence in sentences:
                    sentence.split()
                    if(sentence!=' '):
                        print('insert sentence at newspaper %s'%newspaper_id)
                        command = "insert into sentences (newspaper_id,content) values(%s,%s)"
                        cursor.execute(command,(newspaper_id,sentence))
    connection.commit()
finally:
    connection.close()    

                            