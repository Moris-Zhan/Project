#%%
# https://blog.csdn.net/yoyocat915/article/details/80580066

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait

from bs4 import BeautifulSoup
import requests as req
from MongoDB import MongoDB
import sys

from log import Log
from dateutil import parser
import time
import datetime

headers ={'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) \
                         Chrome/51.0.2704.63 Safari/537.36'}
import re

FlagTimeBreak = False

global logger  
LogObj = Log('Economy')
logger = LogObj.logger

class Crawler:   

    def __init__(self):
        print("Connect to MongoDB")
        self.mongo = MongoDB()
        self.mongo.conn_db(db_name='News')  
        
        self.urlSource={
            # # 匯率新聞 
            # 'Currency':{'USD/JPY':'https://www.investing.com/currencies/usd-jpy-news/',
            #             'EUR/JPY':'https://www.investing.com/currencies/eur-jpy-news/'
            #            },
            # # 貿易商品
            # 'Commodities':{'CurrencyPair':'https://www.investing.com/news/commodities-news/'},
            # 商業政策
            'Economy':{'CurrencyPair':'https://www.investing.com/news/economic-indicators/'}
        }

    def decDate(self,dateTime):
        if type(dateTime) == list:
            if len(dateTime)>0:
                dateTime = dateTime[-1]
        dateTime = parser.parse(dateTime)
        return dateTime
    
    ##替換常用HTML字元實體.
	#使用正常的字元替換HTML中特殊的字元實體.
	#你可以添加新的實體字元到CHAR_ENTITIES中,處理更多HTML字元實體.
	#@param htmlstr HTML字串.
    def replaceCharEntity(self,htmlstr):
        CHAR_ENTITIES={'nbsp':' ','160':' ',
                    'lt':'<','60':'<',
                    'gt':'>','62':'>',
                    'amp':'&','38':'&',
                    'quot':'"','34':'"',}

        re_charEntity=re.compile(r'&#?(?P<name>\w+);')
        sz=re_charEntity.search(htmlstr)
        while sz:
            entity=sz.group()#entity全稱，如&gt;
            key=sz.group('name')#去除&;後entity,如&gt;為gt
            try:
                htmlstr=re_charEntity.sub(CHAR_ENTITIES[key],htmlstr,1)
                sz=re_charEntity.search(htmlstr)
            except KeyError:
                #以空串代替
                htmlstr=re_charEntity.sub('',htmlstr,1)
                sz=re_charEntity.search(htmlstr)
        return htmlstr
	
    def repalce(self,s,re_exp,repl_string):
	    return re_exp.sub(repl_string,s)

    def filter_tags(self,htmlstr):
	    #先過濾CDATA
	    re_cdata=re.compile('//<!\[CDATA\[[^>]*//\]\]>',re.I) #匹配CDATA
	    re_script=re.compile('<\s*script[^>]*>[^<]*<\s*/\s*script\s*>',re.I)#Script
	    re_style=re.compile('<\s*style[^>]*>[^<]*<\s*/\s*style\s*>',re.I)#style
	    re_br=re.compile('<br\s*?/?>')#處理換行
	    re_h=re.compile('</?\w+[^>]*>')#HTML標籤
	    re_comment=re.compile('<!--[^>]*-->')#HTML注釋
	    s=re_cdata.sub('',htmlstr)#去掉CDATA
	    s=re_script.sub('',s) #去掉SCRIPT
	    s=re_style.sub('',s)#去掉style
	    s=re_br.sub('\n',s)#將br轉換為換行
	    s=re_h.sub('',s) #去掉HTML 標籤
	    s=re_comment.sub('',s)#去掉HTML注釋
	    #去掉多餘的空行
	    blank_line=re.compile('\n+')
	    s=blank_line.sub('\n',s)
	    s= self.replaceCharEntity(s)#替換實體
	    return s

    def crawNews(self,tag,currencyPair,currencyPairUrl):        

        driver = webdriver.Chrome(".\chromedriver.exe")
        i = 1
        while(True):
            try:                
                driver.get(currencyPairUrl+'{}'.format(i))
                logger.info(currencyPair + "分頁" + '{}'.format(i))
                
                divNews = driver.find_elements_by_class_name('textDiv')
                del(divNews[0:3]) # 去頭
                del(divNews[10:]) # 去尾

                for idx,div in enumerate(divNews):
                    URLFlagError = False
                    IngoreFlag = False
                    data_dict = None
                    source = None
                    try:
                        headline = div.find_element_by_class_name('title').text
                    except Exception as e:
                        headline = None
                    try:
                        source = div.find_element_by_tag_name('span').get_attribute('innerHTML')
                        source = self.filter_tags(source)
                        source = source.replace('By ','')
                        if '-' in source:
                            source = source.split('-')[0]    
                            source = source.replace(' ','')  
                    except Exception as e:
                        source = None

                    try:         
                        if div.find_element_by_class_name('sponsoredBadge'):
                            IngoreFlag = True
                    except Exception as e:
                        IngoreFlag = False

                    try:
                        # 分頁開啟
                        newsUrl = div.find_element_by_tag_name('a').get_attribute('href')
                        script = "window.open('{}', 'new_window[{}]')".format(newsUrl,1) #開啟分頁語法
                        driver.execute_script(script) #執行分頁
                        driver.switch_to_window(driver.window_handles[1])  #切換瀏覽器索引標籤1                 
                        
                        # Currency
                        if source=='Dailyfx':
                            try:
                                result = req.get(newsUrl, headers=headers)
                                # soup = BeautifulSoup(result.text, "html.parser")
                                pattern = re.compile(r"[\d]+-[\d]+-[\d]+[\w]+:[\d]+" )
                                dateTime = re.findall(pattern,result.text) # 2019-04-15T11:00
                                content = driver.find_element_by_class_name('story_paragraph').get_attribute('innerHTML')
                            except Exception as e:
                                traceback = sys.exc_info()[2]
                                # print(traceback.tb_lineno)
                                # print(e)
                                print("Element Error : Line:{} \n Source:{} \n URL:{}".format(traceback.tb_lineno,source,newsUrl))                   
                        elif source=='FXStreet':
                            try:
                                result = req.get(newsUrl, headers=headers)
                                # soup = BeautifulSoup(result.text, "html.parser")
                                pattern = re.compile(r"[\d]+-[\d]+-[\d]+[\w]+:[\d]+")
                                dateTime = re.findall(pattern,result.text) # 2019-04-10T13:39
                                content = driver.find_element_by_class_name('fxs_article').get_attribute('innerHTML')
                            except Exception as e:
                                traceback = sys.exc_info()[2]
                                # print(traceback.tb_lineno)
                                # print(e)
                                print("Element Error : Line:{} \n Source:{} \n URL:{}".format(traceback.tb_lineno,source,newsUrl))                        
                        elif source=='Forexlive':
                            try:
                                result = req.get(newsUrl, headers=headers)
                                # soup = BeautifulSoup(result.text, "html.parser")
                                pattern = re.compile(r"[\d]+-[\d]+-[\d]+[\w]+:[\d]+")
                                dateTime = re.findall(pattern,result.text) # 2019-04-15T10:51
                                article = driver.find_element_by_tag_name('article')
                                content = article.find_element_by_class_name('artbody').get_attribute('innerHTML')
                            except Exception as e:
                                traceback = sys.exc_info()[2]
                                # print(traceback.tb_lineno)
                                # print(e)
                                print("Element Error : Line:{} \n Source:{} \n URL:{}".format(traceback.tb_lineno,source,newsUrl))
                        elif source=='Marketpulse':
                            try:
                                result = req.get(newsUrl, headers=headers)
                                # soup = BeautifulSoup(result.text, "html.parser")
                                pattern = re.compile(r"[\d]+-[\d]+-[\d]+[\w]+:[\d]+")
                                dateTime = re.findall(pattern,result.text) # 2019-04-15T00:02:51-04:00
                                content = driver.find_element_by_class_name('contents').get_attribute('innerHTML')
                            except Exception as e:
                                traceback = sys.exc_info()[2]
                                # print(traceback.tb_lineno)
                                # print(e)
                                print("Element Error : Line:{} \n Source:{} \n URL:{}".format(traceback.tb_lineno,source,newsUrl))       
                        elif source=='FXEmpire':
                            try:
                                result = req.get(newsUrl, headers=headers)
                                # soup = BeautifulSoup(result.text, "html.parser")
                                pattern = re.compile(r"[\d]+-[\d]+-[\d]+[\w]+:[\d]+")
                                dateTime = re.findall(pattern,result.text) # 2019-04-15T16:30:24
                                content = driver.find_element_by_class_name('Post__PostArticle-sc-1czihrw-0').get_attribute('innerHTML')
                            except Exception as e:
                                traceback = sys.exc_info()[2]
                                # print(traceback.tb_lineno)
                                # print(e)
                                print("Element Error : Line:{} \n Source:{} \n URL:{}".format(traceback.tb_lineno,source,newsUrl))
                        elif source=='MarketWatch': 
                            try:
                                result = req.get(newsUrl, headers=headers)
                                # soup = BeautifulSoup(result.text, "html.parser")
                                pattern = re.compile(r"[\d]+-[\d]+-[\d]+[\w]+:[\d]+")
                                dateTime = re.findall(pattern,result.text)[0] # 2019-04-15T16:30:24
                                content = driver.find_element_by_id('article-body').get_attribute('innerHTML')
                            except Exception as e:
                                traceback = sys.exc_info()[2]
                                # print(traceback.tb_lineno)
                                # print(e)
                                print("Element Error : Line:{} \n Source:{} \n URL:{}".format(traceback.tb_lineno,source,newsUrl))        
                        elif 'Reuters' in source:
                            try:
                                source = 'Reuters'
                                result = req.get(newsUrl, headers=headers)
                                # soup = BeautifulSoup(result.text, "html.parser")
                                pattern = re.compile(r"[\d]+-[\d]+-[\d]+[\w]+:[\d]+")                 

                                contentSectionDetails = driver.find_element_by_class_name('contentSectionDetails')
                                dateTime = contentSectionDetails.find_element_by_tag_name('span').text # 'Aug 13, 2018 04:02AM ET'
                                content = driver.find_element_by_class_name('articlePage').get_attribute('innerHTML')
                                if '(' in dateTime:
                                    dateTime = dateTime.split('(')[1]
                                    dateTime = dateTime.split(')')[0]
                                dateTime = str(parser.parse(dateTime)) 
                            except Exception as e:
                                traceback = sys.exc_info()[2]
                                # print(traceback.tb_lineno)
                                # print(e)
                                print("Element Error : Line:{} \n Source:{} \n URL:{}".format(traceback.tb_lineno,source,newsUrl))
                        elif 'Investing' in source:
                            try:
                                source = 'Investing'
                                result = req.get(newsUrl, headers=headers)
                                # soup = BeautifulSoup(result.text, "html.parser")                                          
                                contentSectionDetails = driver.find_element_by_class_name('contentSectionDetails')
                                date = contentSectionDetails.find_element_by_tag_name('span').text # 'Aug 13, 2018 04:02AM ET'
                                content = driver.find_element_by_class_name('articlePage').get_attribute('innerHTML')
                                
                                if '(' in dateTime:
                                    dateTime = dateTime.split('(')[1]
                                    dateTime = dateTime.split(')')[0]
                                    dateTime = str(parser.parse(dateTime)) 
                            except Exception as e:
                                traceback = sys.exc_info()[2]
                                # print(traceback.tb_lineno)
                                # print(e)
                                print("{} Error , Line:{} URL:{}".format(source,traceback.tb_lineno,newsUrl)) 
                        elif 'Talkmarkets' in source:
                            try:
                                source = 'Talkmarkets'
                                result = req.get(newsUrl, headers=headers)
                                # soup = BeautifulSoup(result.text, "html.parser")
                                pattern = re.compile(r"[\d]+-[\d]+-[\d]+[\w]+:[\d]+")                 

                                dateTime = driver.find_element_by_class_name('text-muted').text # Tuesday, March 19, 2019 4:30 AM EDT
                                content = driver.find_element_by_class_name('tm-article_card-block').get_attribute('innerHTML')
                                
                            except Exception as e:
                                traceback = sys.exc_info()[2]
                                # print(traceback.tb_lineno)
                                # print(e)
                                print("Element Error : Line:{} \n Source:{} \n URL:{}".format(traceback.tb_lineno,source,newsUrl))
                        elif 'Bloomberg' in source:
                            try:
                                source = 'Bloomberg'
                                result = req.get(newsUrl, headers=headers)
                                # soup = BeautifulSoup(result.text, "html.parser")
                                pattern = re.compile(r"[\d]+-[\d]+-[\d]+[\w]+:[\d]+")                 

                                contentSectionDetails = driver.find_element_by_class_name('contentSectionDetails')
                                dateTime = contentSectionDetails.find_element_by_tag_name('span').text # 'Aug 13, 2018 04:02AM ET'
                                content = driver.find_element_by_class_name('articlePage').get_attribute('innerHTML')
                                
                            except Exception as e:
                                traceback = sys.exc_info()[2]
                                # print(traceback.tb_lineno)
                                # print(e)
                                print("Element Error : Line:{} \n Source:{} \n URL:{}".format(traceback.tb_lineno,source,newsUrl))                    
                        elif 'ETF' in source:
                            try:
                                source = 'ETF'
                                result = req.get(newsUrl, headers=headers)
                                # soup = BeautifulSoup(result.text, "html.parser")
                                pattern = re.compile(r"[\d]+-[\d]+-[\d]+[\w]+:[\d]+" )
                                dateTime = re.findall(pattern,result.text) # 2019-04-19T05:30:48-07:00                
                                content = driver.find_element_by_tag_name('article').get_attribute('innerHTML')
                                print()                         
                            except Exception as e:
                                traceback = sys.exc_info()[2]
                                # print(traceback.tb_lineno)
                                # print(e)
                                print("Element Error : Line:{} \n Source:{} \n URL:{}".format(traceback.tb_lineno,source,newsUrl))
                        elif 'Oilprice' in source:
                            try:
                                source = 'Oilprice'
                                result = req.get(newsUrl, headers=headers)
                                # soup = BeautifulSoup(result.text, "html.parser")
                                pattern = re.compile(r"[\d]+-[\d]+-[\d]+[\w]+:[\d]+" )
                                dateTime = re.findall(pattern,result.text) # 2019-04-19T11:00:00-05:00               
                                content = driver.find_element_by_id('article-content').get_attribute('innerHTML')
                                print()                         
                            except Exception as e:
                                traceback = sys.exc_info()[2]
                                # print(traceback.tb_lineno)
                                # print(e)
                                print("Element Error : Line:{} \n Source:{} \n URL:{}".format(traceback.tb_lineno,source,newsUrl))
                        elif 'Hellenic' in source:
                            try:
                                source = 'Hellenic'
                                result = req.get(newsUrl, headers=headers)
                                # soup = BeautifulSoup(result.text, "html.parser")
                                pattern = re.compile(r"[\d]+-[\d]+-[\d]+" )
                                dateTime = re.findall(pattern,result.text) # 2019-04-18              
                                content = driver.find_element_by_class_name('post-inner').get_attribute('innerHTML')
                                print()                         
                            except Exception as e:
                                traceback = sys.exc_info()[2]
                                # print(traceback.tb_lineno)
                                # print(e)
                                print("Element Error : Line:{} \n Source:{} \n URL:{}".format(traceback.tb_lineno,source,newsUrl))
                        elif 'Zacks' in source:
                            try:
                                source = 'Zacks'
                                result = req.get(newsUrl, headers=headers)
                                # soup = BeautifulSoup(result.text, "html.parser")
                                pattern = re.compile(r"[\d]+-[\d]+-[\d]+[\w]+:[\d]+" )
                                dateTime = re.findall(pattern,result.text) # 2019-04-19T17:12:52             
                                content = driver.find_element_by_class_name('commentary_body').get_attribute('innerHTML')
                                print()                         
                            except Exception as e:
                                traceback = sys.exc_info()[2]
                                # print(traceback.tb_lineno)
                                # print(e)
                                print("Element Error : Line:{} \n Source:{} \n URL:{}".format(traceback.tb_lineno,source,newsUrl))                                             
                        # others
                        else:                      
                            try:
                                result = req.get(newsUrl, headers=headers)                            
                                # soup = BeautifulSoup(result.text, "html.parser")
                                pattern = re.compile(r"[\d]+-[\d]+-[\d]+[\w]+:[\d]+")                
                                contentSectionDetails = driver.find_element_by_class_name('contentSectionDetails')
                                dateTime = contentSectionDetails.find_element_by_tag_name('span').text # 'Aug 13, 2018 04:02AM ET'
                                content = driver.find_element_by_class_name('articlePage').get_attribute('innerHTML')                        
                            except Exception as e:
                                traceback = sys.exc_info()[2]     
                                if not IngoreFlag:
                                    logger.error("新新聞來源")
                                    logger.error("{} 新新聞來源 , Line:{} URL : {}".format(source,traceback.tb_lineno,newsUrl))                         

                        driver.close()
                        driver.switch_to_window(driver.window_handles[0])  #切換瀏覽器索引標籤0

                        # 今天日期
                        today = datetime.date.today()
                        today = parser.parse(str(today)) # datetime(UTS)

                        # 資料日期
                        dateTime = self.decDate(dateTime) # datetime(UTS)

                        # 大於2天退出
                        global FlagTimeBreak
                        FlagTimeBreak = False
                        # if (today - dateTime).days > 2:
                        #     FlagTimeBreak = True
                        #     break
                        
                        dateTime = str(dateTime)
                        if not IngoreFlag:
                            # decide currency pair
                            # print(tag + ' : ' + headline)
                            try:
                                logger.info("Write Source:{} , Time : {}".format(source,dateTime))
                                # if currencyPair == 'CurrencyPair':
                                #     logger.info('HeadLine : {}'.format(headline))
                            except Exception as e:
                                print()
                            # change CurrencyPair
                        try:
                            try:
                                data_dict = {
                                    'Tag':tag,
                                    'CurrencyPair':currencyPair,
                                    'Source':source,
                                    'Url':newsUrl,
                                    'HeadLine':headline,                    
                                    'Content': self.filter_tags(content),
                                    'ReleaseTime': dateTime
                                }
                            except Exception as e:
                                # Dailyfx :PageNotFound
                                if not IngoreFlag:
                                    print('Error:{} , URL:{}'.format(source,newsUrl))
                                    logger.error('Error:{} , URL:{}'.format(source,newsUrl))
                                    data_dict = None
                            # print(source,dateTime,sep=' ')                            
                            
                            global WriteDataNum
                            if WriteDataNum > 0 and WriteDataNum %1000 == 0:
                                print('write {} success'.format(WriteDataNum))  
                            if not IngoreFlag:
                                # 檢查已存在資料庫
                                if not self.mongo.searchInDB(data_dict):
                                    self.mongo.insert(data_dict)
                                    WriteDataNum = WriteDataNum + 1
                                else:
                                    logger.info('Already in DataBase: {}'.format(self.mongo.searchInDB(data_dict))) 
                                    logger.info(dateTime) 
                              
                        except Exception as e:
                            traceback = sys.exc_info()[2]
                            logger.error(traceback.tb_lineno)                     
                            logger.error(e) 
                            logger.error("Write Data Error")   
                    except Exception as e:
                        traceback = sys.exc_info()[2]
                        logger.error(traceback.tb_lineno)                     
                        logger.error(e) 
                        URLFlagError = True    
                    
                #     # No Analysis Found  
                logger.info(currencyPair + "分頁" + '{} 爬取完成 {} 筆資料'.format(i,WriteDataNum))
                        
                i = i + 1
                nextUrl = currencyPairUrl+'{}'.format(i)
                result = req.get(nextUrl, headers=headers)
                if 'No Analysis Found' in result.text:
                    logger.info('所有新聞分頁搜尋完畢')
                    logger.info('跳出分頁迴圈')
                    driver.quit()
                    break

                if FlagTimeBreak:
                    logger.info('已爬取近期2天資料')
                    logger.info('跳出分頁迴圈')
                    driver.quit()
                    break
            except Exception as e:
                continue

                
            

                          

s = Crawler()
ds = s.urlSource
for tag in ds.keys():
    global WriteDataNum            
    FlagTimeBreak = False
    pair = ds[tag]
    for p in pair.keys():
        url = pair[p]        
        logger.info("Tag: {} , URL:{}".format(tag,url))
        WriteDataNum = 0
        s.crawNews(tag,p,url)
        logger.info("Tag:{} Pair:{} 成功爬取 {} 新聞資料".format(tag,p,WriteDataNum))
        logger.info(s.mongo.searchInDBCount({'Tag':tag})) 

LogObj.closeLog()

#%% 
#%%  
