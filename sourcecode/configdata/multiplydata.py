import pymysql.cursors

dbname = 'AliceII'
connection = pymysql.connect(host='localhost',
                            user='root',
                            password='12345678',
                            db= dbname,
                            charset='utf8',
                            cursorclass=pymysql.cursors.DictCursor)
def multiply_data(label,number):
    try:
        with connection.cursor() as cursor:
            sql = 'select * from sentences where label = %s limit 600 '
            command = 'insert into sentences (newspaper_id,content,label) values (%s,%s,%s)'
            cursor.execute(sql,label)
            results = cursor.fetchall()
            time = int(int(number)/len(results))
            mod = int(number)%len(results)
            count = 0 
            i = 0
            j = 0 
            while i < time:
                i=i+1
                for result in results:
                    _content = result.get('content')
                    _newspaper_id = result.get('newspaper_id')
                    # print('insert into sentences (newspaper_id,content) values (%s,%s)'%(_newspaper_id,_content))
                    cursor.execute(command,(_newspaper_id,_content,label))
                    count = count +1
            
            while j < mod:
                result = results[j]
                count = count +1
                _content = result.get('content')
                _newspaper_id = result.get('newspaper_id')
                # print('insert into sentences (newspaper_id,content) values (%s,%s)'%(_newspaper_id,_content))
                cursor.execute(command,(_newspaper_id,_content,label))
                j = j +1

            print(time)
            print(mod)
            print(count)
        connection.commit()
    finally:
        connection.close()    

if __name__  == "__main__":
    label = input("label:")
    number = input("number:")
    multiply_data(label,number)