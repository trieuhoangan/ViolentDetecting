import pymysql.cursors
import subprocess
import os
import sys
folder_path = r'sentences'

def check_label(label,limitnum):
    tokenizer_path = r'D:\SkyLabDocument\JVnTextPro\JVnTextPro-3.0.3-executable.jar'
    connection = pymysql.connect(host='localhost',
                                user='root',
                                password='12345678',
                                db='AliceII',
                                charset='utf8',
                                cursorclass=pymysql.cursors.DictCursor)
    try:
        with connection.cursor() as cursor:
            sql = 'select * from sentences where label = %s limit %s'
            cursor.execute(sql,(label,limitnum))
            results = cursor.fetchall()
            for result in results:
                id = str(result.get('id'))
                content = str(result.get('content'))
                content = content.replace('_',' ')
                filename = 'sentences\%s.txt'%(id)
                with open(filename,'w',encoding = 'utf-8',errors ='replace') as f:
                    f.write(content)
                
            subprocess.call(['java', '-jar', tokenizer_path,'-wordseg','-input',folder_path])    
            # for fname in sorted(os.listdir(folder_path)):
            #     filename = os.path.join(folder_path,fname)
            #     id = 
                
            #     with open(filename,'r',encoding = 'utf-8',errors ='replace') as f:
            #         content = f.read()
            #     sql = 'update newspaper set content = "%s" where id = %s'%(content,id)   
            #     cursor.execute(sql) 

            #     filename = filename+'.pro'
        connection.commit()
    finally:
        connection.close()
def check_all_sentences():
    tokenizer_path = r'D:\SkyLabDocument\JVnTextPro\JVnTextPro-3.0.3-executable.jar'
    connection = pymysql.connect(host='localhost',
                                user='root',
                                password='12345678',
                                db='AliceII',
                                charset='utf8',
                                cursorclass=pymysql.cursors.DictCursor)
    try:
        with connection.cursor() as cursor:
            sql = 'select * from sentences'
            cursor.execute(sql)
            results = cursor.fetchall()
            for result in results:
                id = str(result.get('id'))
                content = str(result.get('content'))
                content = content.replace('_',' ')
                filename = 'sentences\%s.txt'%(id)
                with open(filename,'w',encoding = 'utf-8',errors ='replace') as f:
                    f.write(content)
                
            subprocess.call(['java', '-jar', tokenizer_path,'-wordseg','-input',folder_path])    
            # for fname in sorted(os.listdir(folder_path)):
            #     filename = os.path.join(folder_path,fname)
            #     id = 
                
            #     with open(filename,'r',encoding = 'utf-8',errors ='replace') as f:
            #         content = f.read()
            #     sql = 'update newspaper set content = "%s" where id = %s'%(content,id)   
            #     cursor.execute(sql) 

            #     filename = filename+'.pro'
        connection.commit()
    finally:
        connection.close()
def update_database():
    connection = pymysql.connect(host='localhost',
                                user='root',
                                password='12345678',
                                db='AliceII',
                                charset='utf8',
                                cursorclass=pymysql.cursors.DictCursor)

    folder_path = r'sentences'
    try:
        with connection.cursor() as cursor:
            for fname in sorted(os.listdir(folder_path)):
                    filename = "sentences/%s"%(fname)
                    
                    if '.pro' in fname:
                        ids = fname.split('.')
                        id = ids[0]
                        # print(filename)
                        print(id)
                        with open(filename,'r',encoding = 'utf-8',errors ='replace') as f:
                        #     print(id)
                            content = f.read()
                            # valid_file_name_character = re.compile('[\\~#%&*{}/:<>?|\"-]')
                            # content = re.sub(valid_file_name_character,'',content)
                            content = content.replace('\xa0',' ')
                            # print(content)
                            sql = 'update sentences set content ="%s" where id=%s'
                            # print(ss--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
                            cursor.execute(sql,(content,id))
        connection.commit()
    finally:
        connection.close()
check_label(0,2000)
check_label(1,2000)
check_label(2,2000)
# check_all_sentences()
update_database()


