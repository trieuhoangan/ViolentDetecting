import scrapy
import re
from scrapy.selector import Selector
from scrapy.http import HtmlResponse
class UpdateSpider(scrapy.Spider):
    name = 'update'
    def  start_requests(self):
        urls = [
            'https://thuvienphapluat.vn/phap-luat/tim-van-ban.aspx?keyword=&match=True&area=0#'
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)
    def parseUpdate(self, response):
        openupdatefile = open('update.txt','r+')
        updateline = openupdatefile.readlines
        listofVB = response.css('p.nqTitle a::attr(href)').extract()
        if(updateline != listofVB[0]):
            openupdatefile.write(str(listofVB[0]))
            for VB in listofVB:
                if VB != updateline:
                    yield scrapy.Request(url=VB, callback = self.parse)
        
