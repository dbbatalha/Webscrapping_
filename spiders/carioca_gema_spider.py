# -*- coding: utf-8 -*-
import scrapy
from ..items import CariocaGemaItem



class CariocaGemaSpiderSpider(scrapy.Spider):
    name = 'carioca_gema_spider'
    allowed_domains = ['tripadvisor.com']
    start_urls = ['http://www.tripadvisor.com/Attraction_Review-g303506-d2467603-Reviews-or10-Carioca_da_Gema-Rio_de_Janeiro_State_of_Rio_de_Janeiro.html/']

    def parse(self, response):
        item = CariocaGemaItem()
        quadros_de_comentarios = response.xpath("//div[@class='bPhtn']/div")
        for quadro in quadros_de_comentarios:
            item["autor_comentario"] = quadro.xpath(".//div[@class='dDwZb f M k']//div[@class='cjhIj']//span[@class='WlYyy cPsXC dTqpp']//a[@target='_self']/text()").get() #.// é o ponto duas barras que assegura que o meu robo irá para o proximo item

            item["autor_endereco"] = quadro.xpath(".//div[@class='dDwZb f M k']//div[@class='cjhIj']//div[@class='ddOtn']//div[@class='WlYyy diXIH bQCoY']//span/text()").get()

            item["comentario_titulo"] = quadro.xpath(".//div[@class='WlYyy cPsXC bLFSo cspKb dTqpp']//span[@class='NejBf']/text()").get()

            item["comentario_corpo"] = quadro.xpath(".//div[@class='WlYyy diXIH dDKKM']//span[@class='NejBf']/text()").get()

            item["comentario_nota"] = quadro.xpath(".//span[@data-ft='true']//svg[@class='RWYkj d H0']/@title").get()

            item["comentario_data"] = quadro.xpath(".//div[@class='fEDvV']/text()").get()
            yield item
        
        next_page = response.xpath("//a[@aria-label='Next page']/@href").get()#é o /@href que assegura que o robo irá acessar o link
        #  @class='dfuux f u j _T z _F _S ddFHE bVTsJ emPJr' and @aria-label='Próxima página
        if next_page:
            yield response.follow(url=next_page, callback=self.parse)

# https://www.tripadvisor.com/Restaurant_Review-g303506-d2235233-Reviews-Fogo_de_Chao-Rio_de_Janeiro_State_of_Rio_de_Janeiro.html (proximo projeto)