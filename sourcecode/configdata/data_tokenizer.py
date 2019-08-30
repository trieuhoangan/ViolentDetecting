
import pymysql.cursors
import subprocess
import os
import sys

folder_path = r'newspapers'

tokenizer_path = r'D:\SkyLabDocument\JVnTextPro\JVnTextPro-3.0.3-executable.jar'
connection = pymysql.connect(host='localhost',
                            user='root',
                            password='12345678',
                            db='AliceII',
                            charset='utf8',
                            cursorclass=pymysql.cursors.DictCursor)
try:
    with connection.cursor() as cursor:
        sql = 'select * from newspaper where id not in (select distinct newspaper_id from sentences )'
        cursor.execute(sql)
        results = cursor.fetchall()
        for result in results:
            id = str(result.get('id'))
            content = str(result.get('content'))
            content = content.replace('_',' ')
            filename = 'newspapers\%s.txt'%(id)
            with open(filename,'w',encoding = 'utf-8',errors ='replace') as f:
                f.write(content)
            
        subprocess.call(['java', '-jar', tokenizer_path, '-senseg','-wordseg','-input',folder_path])    
       
    connection.commit()
finally:
    connection.close()
   
