import pymysql.cursors

dbname = 'aliceii'
sql = 'select * from tmp'
connection = pymysql.connect(host='localhost',
                            user='root',
                            password='12345678',
                            db=dbname,
                            charset='utf8',
                            cursorclass=pymysql.cursors.DictCursor)
try:
    with connection.cursor() as cursor:
        cursor.execute(sql)
        results = cursor.fetchall()
        for result in results:
            _id = str(result.get('id'))
            _label = str(result.get('label'))
            # print(_id+' '+_label)
            command = "update sentences set label = %s where id = %s"
            cursor.execute(command,(_label,_id))
            print(command)
    connection.commit()
finally:
    connection.close()
