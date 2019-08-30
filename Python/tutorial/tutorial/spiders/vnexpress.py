# -*- coding: utf-8 -*-
import scrapy
import re
import pymysql.cursors
from scrapy.selector import Selector
from scrapy.http import HtmlResponse
from tutorial.customlib import accessMysql

class VnexpressSpider(scrapy.Spider):
    name = 'vnexpress'
    # allowed_domains = ['https://vnexpress.net/']
    start_urls = ['https://vnexpress.net/phap-luat']
    host = 'localhost'
    user = 'root'
    password = '12345678'
    db = 'aliceii'
    dataTable = 'newspaper'
    checkTable = "violentCriteria"
    folder = 'vnexpress'
    def parse(self, response):
        #crawl each newspaper in this page
        section_sidebar1 = response.css('section.container section.sidebar_1')
        articles = section_sidebar1.css('article')
        for article in articles:
            link = article.css('h4 a::attr(href)').extract_first()
            if 'phap-luat' in link:
                yield response.follow(link,self.parse_VB)

        #crawl to other pages
        Pages = response.css('div[id*=pagination] a::attr(href)').extract()
        if len(Pages)==6:
            nextPage = Pages[5]
            if "http" not in nextPage:
                nextPage = "https://vnexpress.net"+nextPage
            yield response.follow(nextPage,callback = self.parse)
        else:
            nextPage = Pages[6]
            if "http" not in nextPage:
                nextPage = "https://vnexpress.net"+nextPage
            yield response.follow(nextPage,callback = self.parse)

    def parse_VB(self, response):
        section = response.css("section.container section.wrap_sidebar_12")
        VB_container = section.css('section.sidebar_1')[0]
        if bool(VB_container)==False:
            VB_container = response.css("div.fck_detail")
        #extract content
        tag_ps = VB_container.css('p').extract()
        content = ' '
        for tag_p in tag_ps:
            content = content + str(tag_p)
        vb = accessMysql.normalizeContent(content)
        title = response.css("h1.title_news_detail::text").extract_first()  
        title = accessMysql.normalizeTitle(title)
        if accessMysql.isViolent(vb,self.host,self.user,self.password,self.db,self.checkTable):
            print("saving document in sql.......................")
            accessMysql.saveContentIntoNewspaperTable(self.host,self.user,self.password,self.db,title,vb)
            print("saving document in text.......................")
            accessMysql.save_content_to_file(vb,title,self.folder)
