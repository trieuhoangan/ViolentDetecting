# -*- coding: utf-8 -*-
import scrapy
from tutorial.customlib import accessMysql
i=0
j =0 
limit = 3000
class DocbaovnSpider(scrapy.Spider):
    name = 'docbaovn'
    # allowed_domains = ['http://docbao.vn']
    start_urls = ['http://docbao.vn/phap-luat','http://docbao.vn/xa-hoi',]
    host = 'localhost'
    user = 'root'
    password = '12345678'
    db = 'aliceii'
    dataTable = 'newspaper'
    checkTable = 'violentCriteria'
    folder = 'docbaovn'
    domain = 'http://docbao.vn'
    def parse(self, response):
        global j
        if i<limit:
            news = response.css('div.category_news_list ul li')
            for new in news:
                link = new.css('a::attr(href)').extract_first()
                link = self.domain+link
                yield response.follow(link,self.parseContent)

            pages = response.css('div.page-pagination a::attr(href)').extract()
            if len(pages)==1 and i==0 and j<500:
                j = j+1
                yield response.follow(self.domain+pages[0],self.parse)
            else:
                if j < 5000:
                    j = j+1
                    yield response.follow(self.domain+pages[1],self.parse)
                
    def parseContent(self, response):        
        title  = response.css('div.detail_top h1 span::text').extract_first()
        content = response.css('div[id*=detail_ct]').extract_first()
        global i
        vb = accessMysql.normalizeContent(content)
        title = accessMysql.normalizeTitle(title)
        if accessMysql.isViolent(vb,self.host,self.user,self.password,self.db,self.checkTable):
            print("saving document in sql.......................")
            accessMysql.saveContentIntoNewspaperTable(self.host,self.user,self.password,self.db,title,vb)
            print("saving document in text.......................")
            accessMysql.save_content_to_file(vb,title,self.folder)
        i=i+1
