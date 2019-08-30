# -*- coding: utf-8 -*-
import scrapy
import re
import pymysql.cursors
from tutorial.customlib import accessMysql
i=0
j=0
limit=1000
class TinmoiSpider(scrapy.Spider):
    name = 'tinmoi'
    # allowed_domains = ['http://www.tinmoi.vn']
    start_urls = ['http://www.tinmoi.vn/C/Tin-tuc',
                  'http://www.tinmoi.vn/phap-luat/trong-an',
                  'http://www.tinmoi.vn/phap-luat/vu-an-noi-tieng',  
                  'http://www.tinmoi.vn/phap-luat/an-ninh-hinh-su',
                    ]
    host = 'localhost'
    user = 'root'
    password = '12345678'
    db = 'aliceii'
    dataTable = 'newspaper'
    checkTable = "violentCriteria"
    folder = 'tinmoi'
    def parse(self, response):
        if i<limit:
            listVB = response.css('div[id*=width-480] ul')
            VBs = listVB.css('li')
            for VB in VBs:
                link = VB.css('h3 a::attr(href)').extract_first()
                if link is None:
                    link = VB.css('h4 a::attr(href)').extract_first()
                    if link is not None:
                        yield response.follow(link,self.parseContent)

            pages = response.css('div.paging a::attr(href)').extract()
            
            if len(pages) >7:
                yield response.follow(pages[5],self.parse)    
            else:
                yield response.follow(pages[6],self.parse)

    def parseContent(self, response):
        content = response.css('div[id*=tm-content]')
        global i
        vb = accessMysql.normalizeContent(content.extract_first())
        vb = vb.replace('(adsbygoogle = window.adsbygoogle || []).push({});','')
        title = accessMysql.normalizeTitle(response.css('h1.newstitle::text').extract_first())
        if accessMysql.isViolent(vb,self.host,self.user,self.password,self.db,self.checkTable):
            print("saving document in sql.......................")
            accessMysql.saveContentIntoNewspaperTable(self.host,self.user,self.password,self.db,title,vb)
            print("saving document in text.......................")
            accessMysql.save_content_to_file(vb,title,self.folder)
        i=i+1
