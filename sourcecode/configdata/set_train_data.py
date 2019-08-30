import pymysql.cursors

connection = pymysql.connect(host='localhost',
                                user='root',
                                password='12345678',
                                db='AliceIII',
                                charset='utf8',
                                cursorclass=pymysql.cursors.DictCursor)
try:
    with connection.cursor() as cursor:
        sql = 'select * from sentences where label = 1 limit 1720,100'
        cursor.execute(sql)
        results = cursor.fetchall()
        for result in results:
            _id = result.get('id')
            sql = 'update sentences set isTrain = 2 where id = %s' 
            cursor.execute(sql,(_id))
    connection.commit()
finally:
    connection.close()