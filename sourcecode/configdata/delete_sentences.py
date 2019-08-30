import pymysql.cursors

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
            ids = []
            sql = "select id from sentences limit 0,700000"
            cursor.execute(sql)
            results = cursor.fetchall()
            for result in results:
                _id = str(result.get('id'))
                sql = 'delete from sentences where id =%s'
                cursor.execute(sql,_id)

    connection.commit()
finally:
    connection.close()    