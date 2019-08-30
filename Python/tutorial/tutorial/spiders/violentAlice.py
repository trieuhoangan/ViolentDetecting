# -*- coding: utf-8 -*-
import scrapy
import re
import pymysql.cursors
from scrapy.selector import Selector
from scrapy.http import HtmlResponse
from tutorial.customlib import accessMysql
i = 0
j = 0
limit = 6000
#numbers = open("D:\\Python\tutorial\number.txt",'r')
#i = numbers.read
#numbers.close
#numbers = open("D:\\Python\tutorial\chapter.txt",'r')
#j = numbers.read
#numbers.close

class AliceSpider(scrapy.Spider):
    
    name = "violentalice"
    host = 'localhost'
    user = 'root'
    password = '12345678'
    db = 'alice'
    dataTable = 'newspaper'
    checkTable = "violentCriteria"
    folder = 'tinmoi'
    def start_requests(self):
        urls = [
            'https://vnexpress.net/tin-tuc/thoi-su',
            'https://vnexpress.net/tin-tuc/the-gioi',
            'https://kinhdoanh.vnexpress.net/',
            'https://giaitri.vnexpress.net/',
            'https://thethao.vnexpress.net',
            'https://vnexpress.net/tin-tuc/phap-luat',
            'https://vnexpress.net/tin-tuc/giao-duc',
            'https://suckhoe.vnexpress.net',
            'https://doisong.vnexpress.net/',
            'https://dulich.vnexpress.net',
            'https://vnexpress.net/tin-tuc/khoa-hoc',
            'https://sohoa.vnexpress.net/',
            'https://vnexpress.net/tin-tuc/oto-xe-may',
            'https://vnexpress.net/tin-tuc/cong-dong',
            'https://vnexpress.net/tin-tuc/tam-su',
            'https://vnexpress.net/tin-tuc/cuoi',
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):	
        global limit
        if i < limit:
            listofVB = response.css('h3.title_news a.icon_commend::attr(href)').extract()
            Pages = response.css('div[id*=pagination] a')
            if len(Pages)==6:
                nextPage = Pages[5].css('a::text').extract_first()
                if bool(nextPage)==False:
                    if "http" not in nextPage:
                        nextPage = "https://vnexpress.net"+nextPage
                    yield response.follow(nextPage,callback = self.parse)
            else:
                nextPage = Pages[4].extract()
            
                if "http" not in nextPage:
                    nextPage = "https://vnexpress.net"+nextPage
                yield response.follow(nextPage,callback = self.parse)

           
        
    def parse_VB(self, response):
        global i
        global limit
        if i < limit:
        #VB_container = response.css('div.cldivContentDocVN').extract()
        #filee = response.css("title::text").extract_first().strip()
        #filename = str(filee)+".txt"
            # a = str(i)
            connection = pymysql.connect(host='localhost',
                             user='root',
                             password='12345678',
                             db='Aliceii',
                             charset='utf8',
                             cursorclass=pymysql.cursors.DictCursor)
            

            try:
                VB_container = response.css("article.content_detail")
                if bool(VB_container)==False:
                    VB_container = response.css("div.fck_detail")
                title = response.css("h1.title_news_detail::text").extract_first()
                cleanr = re.compile('<.*?>',flags=re.DOTALL)
                title = re.sub(cleanr,'',title)
                valid_file_name_character = re.compile('[\\~#%&*{}/:<>?|\"-]')
                title = re.sub(valid_file_name_character,'',title)
                title = title.replace('\n','_')
                title = title.replace(' ','_')
                title = title.strip()
                with connection.cursor() as cursor:
                    sql = "SELECT `title` FROM `newspaper` WHERE `title`=%s"
                    cursor.execute(sql, (title))
                    result = cursor.fetchone()
                    if bool(result)==False:
                    # Create a new record
                        filename ='tinmoi/'+title+".txt"
                        f = open(filename,'w',encoding='utf-8')
                        VB_saver = str(VB_container.extract_first())
                        
                        # n = len(Cat)
                        if accessMysql.isViolent(VB_saver,self.host,self.user,self.password,self.db,self.checkTable):
                            VB_saver = VB_saver.replace('\r','')
                            VB_saver = VB_saver.replace('\xa0','')
                            VB_saver = VB_saver.replace('&nbsp','')
                            VB_saver = VB_saver.strip()
                            print("saving document")
                            Cat = re.sub(cleanr,'',VB_saver)
                            Cat = Cat.strip()
                            f.write(Cat)
                            i = i +1
                            f.close
                            sentences = Cat.split('.')
                            numberOfSentence = int(len(sentences))
                            sql = "INSERT INTO `newspaper` (`title`, `content`,`sentences`) VALUES (%s, %s, %s)"
                            cursor.execute(sql, (title,Cat,numberOfSentence))


                connection.commit()

            finally:
                connection.close()          