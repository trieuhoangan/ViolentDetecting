import pymysql.cursors
import os
import re

connection = pymysql.connect(host='localhost',
                            user='root',
                            password='12345678',
                            db='AliceII',
                            charset='utf8',
                            cursorclass=pymysql.cursors.DictCursor)
try:
    with connection.cursor() as cursor:
        sql = "select id from newspaper where id not in ( select distinct newspaper_id from sentences)"
        cursor.execute(sql)
        un_extracted_newspapers = cursor.fetchall()
        un_extracted_newspaper_id = []
        for un_extracted_newspaper in un_extracted_newspapers:
            un_extracted_newspaper_id.append(str(un_extracted_newspaper.get('id')))
            # print(str(un_extracted_newspaper.get('id')))
        # print(un_extracted_newspaper_id)
        folder_path = r'newspapers'
        id_list = []
        for fname in sorted(os.listdir(folder_path)):
            filename = "newspapers/%s"%(fname)
            
            if '.pro' in fname:
                ids = fname.split('.')
                _id = ids[0]
                id_list.append(_id)
                # print(filename)
                # print(_id)
                with open(filename,'r',encoding = 'utf-8',errors ='replace') as f:
                #     print(_id)
                    sentences = f.readlines()
                    content = f.read()
                    valid_file_name_character = re.compile('[\\~#%&*{}/:<>?|\"-]')
                    content = re.sub(valid_file_name_character,'',content)
                    content = content.replace('\xa0',' ')
                    # print(content)
                    sql = 'update newspaper set content ="%s" where id=%s'
                    # print('update newspaper set content ="%s" where id=%s'%(content,id))
                    # print("----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
                    cursor.execute(sql,(content,_id))

                    # print(sentences)
                    # __id = str(_id)
                    if _id in un_extracted_newspaper_id:
                        # print("get in if")
                        for sentence in sentences:
                            sentence = re.sub(valid_file_name_character,'',sentence)
                            sentence = sentence.replace('\xa0',' ')
                            print("inserting")
                            id_number=int(_id)
                            command = "insert into sentences (newspaper_id,content) values(%s,%s)"
                            # print('insert into sentences("newspaper_id","content") values (%d,%s)'%(int(_id),sentence))
                            cursor.execute(command,(_id,sentence))
        # print(id_list)
    connection.commit()
finally:
    connection.close()

