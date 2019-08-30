import pymysql.cursors
from nltk.tokenize import sent_tokenize

if __name__=='__main__':
    table = 'violentCriteria'
    connection = pymysql.connect(host = 'localhost',user = 'root', password = '12345678', db = 'aliceii', charset="utf8",cursorclass=pymysql.cursors.DictCursor)
    try:
        with connection.cursor() as cursor:
            # ================================================
            # sql = "SELECT `violentWord` FROM %s" % table
            # cursor.execute(sql)
            # results = cursor.fetchall()
            # for result in results:
            #     print(result) 
            # =================================================
            sql ='select `content` from newspaper where id = 5'
            cursor.execute(sql)
            result = cursor.fetchone()
            content = result.get('content')
            sentences = sent_tokenize(content)
            for sentence in sentences:
                print(sentence)
                print('\n')
    finally:
        connection.close()