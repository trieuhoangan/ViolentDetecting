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
            sql = "SELECT * FROM newspaper"
            cursor.execute(sql)
            results = cursor.fetchall()
            print("start to extract")
            for result in results:
                content = str(result.get('content'))
                content = accessMysql.normalizeContent(content)
                sentences = content.split('\t')
    
    connection.commit()
finally:
    connection.close()    

                            