import pymysql.cursors
from random import randint

connection = pymysql.connect(host='localhost',
                            user='root',
                            password='12345678',
                            db='AliceII',
                            charset='utf8',
                            cursorclass=pymysql.cursors.DictCursor)
try:
    with connection.cursor() as cursor:
        i = 0
        while i<6000:
            sql = "UPDATE `newspaper` set `newspaper`.`label` = %s WHERE `newspaper`.`id` = %s"
            cursor.execute(sql,(randint(1,5),i))
            i=i+1
    connection.commit()
finally:
    connection.close()


