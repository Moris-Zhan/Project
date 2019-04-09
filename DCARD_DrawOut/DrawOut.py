# coding=utf-8

import sys
import time
import datetime as dt
from datetime import datetime
# ------------------------------------------------------------------------------------------
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
# ------------------------------------------------------------------------------------------
import logging
import os
# ------------------------------------------------------------------------------------------
from email.mime.text import MIMEText
# from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
# from smtplib import SMTP
import smtplib
# ------------------------------------------------------------------------------------------
import requests as req
from PIL import Image
from io import BytesIO
# ------------------------------------------------------------------------------------------
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array

global model
import cv2
import dlib
import imutils
from imutils.face_utils import *

import requests as req
from PIL import Image
from io import BytesIO
import numpy as np


# ------------------------------------------------------------------------------------------
def get_face(img):
	global detector, landmark_predictor
	# 宣告臉部偵測器，以及載入預訓練的臉部特徵點模型
	detector = dlib.get_frontal_face_detector()
	landmark_predictor = dlib.shape_predictor('DCARD/shape_predictor_68_face_landmarks.dat')

	# 產生臉部識別
	face_rects = detector(img, 1)
	for i, d in enumerate(face_rects):
		# 讀取框左上右下座標
		x1 = d.left()
		y1 = d.top()
		x2 = d.right()
		y2 = d.bottom()
		# 根據此座標範圍讀取臉部特徵點
		shape = landmark_predictor(img, d)
		# 將特徵點轉為numpy
		shape = shape_to_np(shape)  # (68,2)
		# 透過dlib挖取臉孔部分，將臉孔圖片縮放至256*256的大小，並存放於pickle檔中
		# 人臉圖像部分呢。很簡單，只要根據畫框的位置切取即可crop_img = img[y1:y2, x1:x2, :]
		crop_img = img[y1:y2, x1:x2, :]
		try:
			crop_img = cv2.resize(crop_img, (128, 128))
			return crop_img
		except:
			return np.array([0])
	return np.array([0])


def predict_image(image):
	model = load_model('DCARD/faceRank.h5')
	model.load_weights('DCARD/faceRank_weights.h5')
	opencvImage = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  # predict target
	face = get_face(opencvImage)
	face = face.astype('float32')
	face /= 255
	image = img_to_array(face)
	img = image[np.newaxis, :, :]
	score = model.predict(img)[0][0] * 20
	logger.info("Predict Score : {}".format(score))
	return int(score)


def download_image(url, name, p=True):
	try:
		headers = {
			'user-agent': 'Mozilla/5.0 (Macintosh Intel Mac OS X 10_13_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.181 Safari/537.36'}
		response = req.get(url, headers=headers)
		image = Image.open(BytesIO(response.content))
		image.show()

		date = datetime.now().strftime("%Y-%m-%d")
		img_dir = 'DCARD Image/' + date + '/'
		logger.info(img_dir)

		if not os.path.exists(img_dir):
			logger.info('create folder')
			os.makedirs(img_dir)
		image.save(img_dir + "{}.jpg".format(name.replace('\t', ' ')))

		if p:
			try:
				score = predict_image(image)
				if score > 60:
					return 'beautiful', '#ff1aff', score
				else:
					return 'normal', 'black', score
			except:
				return 'Unknown', 'black', 0
		else:
			return "Unknown", "black", 0
	except Exception as e:
		traceback = sys.exc_info()[2]
		# logger.error(sys.exc_info())
		logger.error(traceback.tb_lineno)
		logger.error(e)


def getDay():
	now = datetime.now()
	today = str(now.strftime("%Y-%m-%d"))

	now -= dt.timedelta(days=1)
	yesterday = "{}月{}日".format(now.month, now.day)
	return today, yesterday


def getLogger(loggerName, loggerPath):
	# 設置logger
	logger = logging.getLogger(loggerName)  # 不加名稱設置root logger
	logger.setLevel(logging.DEBUG)
	formatter = logging.Formatter(
		'%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
		datefmt='%Y-%m-%d %H:%M:%S')
	logging.Filter(loggerName)

	# 使用FileHandler輸出到文件
	directory = os.path.dirname(loggerPath)
	if not os.path.exists(directory):
		os.makedirs(directory)
	fh = logging.FileHandler(loggerPath)

	fh.setLevel(logging.DEBUG)
	fh.setFormatter(formatter)

	# 使用StreamHandler輸出到屏幕
	ch = logging.StreamHandler()
	ch.setLevel(logging.DEBUG)
	ch.setFormatter(formatter)
	# 添加兩個Handler
	logger.addHandler(ch)
	logger.addHandler(fh)
	# Handler只啟動一次
	# 設置logger
	logger.info(u'logger已啟動')
	return logger


def open_web(account, pwd, alis):
	try:
		url = "https://www.dcard.tw/login"
		try:
			driver = webdriver.Chrome("D:\\Program\\Anaconda3\\chromedriver.exe")
			driver.get(url)
		except:
			driver = webdriver.Chrome("E:\\Program\\Anaconda3\\chromedriver.exe")
			driver.get(url)

		# 輸入帳號
		accountBox = driver.find_element_by_xpath(
			'//*[@id="root"]/div/div[1]/div/div/div/div[1]/form/label[1]/input')
		accountBox.send_keys(account)  # 清空內容

		# 輸入密碼
		pwdBox = driver.find_element_by_xpath(
			'//*[@id="root"]/div/div[1]/div/div/div/div[1]/form/label[2]/input')
		pwdBox.send_keys(pwd)  # 清空內容

		driver.find_element_by_xpath('//*[@id="root"]/div/div[1]/div/div/div/div[1]/form/button').click()  # 送出
		time.sleep(5)

		# 前往抽卡頁面
		while (True):
			try:
				locator = (By.XPATH, '//*[@id="root"]/div/header/div/nav/div[3]/a/span')
				WebDriverWait(driver, 5, 0.5).until(EC.presence_of_element_located(locator))
				driver.find_element_by_xpath('//*[@id="root"]/div/header/div/nav/div[3]/a/span').click()
				logger.info('前往抽卡頁面')
				break
			except:
				pass
				driver.refresh()

		# 判斷性別
		locator = (By.XPATH, '//*[@id="root"]/div/div[1]/div/div/div[1]/div/div[2]/div[1]/div[4]/div/div[1]')
		WebDriverWait(driver, 20, 0.5).until(EC.presence_of_element_located(locator))
		sex = driver.find_element_by_xpath(
			'//*[@id="root"]/div/div[1]/div/div/div[1]/div/div[2]/div[1]/div[4]/div/div[1]').text

		mail_text = u'<html><body>'
		box_flag = False
		if u'女同學' in sex:

			locator = (By.XPATH, '//*[@id="root"]/div/div[1]/div/div/div[1]/div/div[2]/div[1]/div[3]/div/div/img')
			WebDriverWait(driver, 20, 0.5).until(EC.presence_of_element_located(locator))
			img_src = driver.find_element_by_xpath(
				'//*[@id="root"]/div/div[1]/div/div/div[1]/div/div[2]/div[1]/div[3]/div/div/img').get_attribute('src')

			locator = (By.XPATH, '//*[@id="root"]/div/div[1]/div/div/div[1]/div/div[2]/div[1]/div[4]/div/div[2]/div')
			WebDriverWait(driver, 20, 0.5).until(EC.presence_of_element_located(locator))
			info = driver.find_element_by_xpath(
				'//*[@id="root"]/div/div[1]/div/div/div[1]/div/div[2]/div[1]/div[4]/div/div[2]/div').text
			info = info.replace(' ', '	-	')
			logger.info(img_src)
			logger.info(info + u" " + sex)
			info = info.replace('＆emsp;', ' ')
			face_text, color, score = download_image(img_src, info)
			mail_text += '<h2>' + alis + '</h2>'
			try:
				# mail_text += '<font size="4" color="' + color + '">' + info + '</font>' + '<br>'
				# mail_text += '<font size="5" color="' + color + '">' + face_text +'</font>'

				mail_text += '<font size="4" color="' + color + '">' + info + '</font>' + '<br>'
				mail_text += '<font size="5" color="' + color + '">' + u"顏質分數 ==> " + str(score) + '</font>' + '<br>'
				mail_text += '<font size="5" color="' + color + '">' + face_text + '</font>' + '<br>'

				# mail_text += '<font size="5" color="' + color + '">' + u"顏質分數 ==> " + score + face_text +'</font>'
				# mail_text += '<font size="5" color="' + color + '">' + " ==> " + face_text + '</font>'
				mail_text += '<img src="'
				# 送出邀請
				driver.find_element_by_xpath(
					'//*[@id="root"]/div/div[1]/div/div/div[1]/div/div[2]/div[1]/div[4]/div/div[3]/button').click()
				# 打招呼
				driver.find_element_by_xpath('//*[@id="root"]/div/div[2]/div/div/div/div/form/textarea').send_keys(
					u'Hi~~妳好')
				# 送出
				driver.find_element_by_xpath('//*[@id="root"]/div/div[2]/div/div/div/div/form/div/button[1]').click()
				logger.info(u'已送出邀請')
				mail_text += img_src + '">' + u'抽卡 - 狀態 : ' + u'已送出邀請' + '</a></h4>'
			except Exception as e:
				logger.info(u'已重複抽卡')
				mail_text += img_src + '">' + u'抽卡 - 狀態 : ' + u'已重複抽卡' + '</a></h4>'
			# traceback = sys.exc_info()[2]
			# logger.error(sys.exc_info())
			# logger.error(traceback.tb_lineno)
			# logger.error(e)
		else:
			mail_text += '<h4>' + alis + '</h4>'
			mail_text += u'<h4>歐歐~~今天抽到的男孩子呢ㅠㅠ 明日請再接再厲:")</h4>'
			logger.info(u'今日的抽卡是男生')

		# time.sleep(3)
		driver.refresh()
		# 點擊通知
		locator = (By.XPATH, '//*[@id="root"]/div/header/div/nav/div[2]/div/div/div/div/div[1]/div')
		WebDriverWait(driver, 20, 0.5).until(EC.presence_of_element_located(locator))
		driver.find_element_by_xpath('//*[@id="root"]/div/header/div/nav/div[2]/div/div/div/div/div[1]/div').click()
		# time.sleep(1)

		locator = (By.CLASS_NAME, 'Notification_body_3EvRye')
		WebDriverWait(driver, 20, 0.5).until(EC.presence_of_element_located(locator))
		notfy_list = driver.find_elements_by_class_name('Notification_body_3EvRye')
		date_list = driver.find_elements_by_class_name('Notification_footer_131P9a')

		if len(notfy_list) > 0:
			for index in range(0, 5):
				nofify = notfy_list[index].text
				date = date_list[index].text
				if (u'命運之神' in nofify) and (date == yesterday):
					logger.info(date + u"恭喜獲得新卡友")
					logger.info(nofify)

					notfy_list[index].click()
					logger.info(u'前往新好友頁面')
					driver.find_element_by_xpath(
						'//*[@id="root"]/div/div[1]/div/div/div/div[2]/div/div[1]/div/div[2]').click()
					logger.info(u'點擊好友關於')
					time.sleep(3)
					img_src = driver.find_element_by_class_name('GalleryImage_avatarImage_hCD2rx').get_attribute('src')
					logger.info(img_src)
					mail_text += u'<h2 align="center">' + date + u'恭喜獲得新卡友</h2><br>'
					mail_text += '<h4><img src="' + img_src + '"></h4>'
					mail_text += u'<h2 align="center">' + nofify + '</h2>'
					box_flag = True
					break
				if index == 4:
					logger.info(u'無新卡友')

		else:
			logger.info(u'無新卡友')

		# 若無新卡友切至收件閘
		if not box_flag:
			logger.info(u'切至收件閘')
			locator = (By.XPATH, '//*[@id="root"]/div/header/div/nav/div[4]/a/span')
			WebDriverWait(driver, 20, 0.5).until(EC.presence_of_element_located(locator))
			driver.find_element_by_xpath('//*[@id="root"]/div/header/div/nav/div[4]/a/span').click()
		# 未讀信件訊息
		# time.sleep(5)

		# 切至未回覆信件閘
		locator = (By.XPATH, '//*[@id="root"]/div/div[1]/div/div/div/div[1]/div[1]/div/div[2]')
		WebDriverWait(driver, 10, 0.5).until(EC.presence_of_element_located(locator))
		driver.find_element_by_xpath('//*[@id="root"]/div/div[1]/div/div/div/div[1]/div[1]/div/div[2]').click()

		# 收集未讀信件條
		msgInfo = driver.find_element_by_xpath('//*[@id="root"]/div/div[1]/div/div/div/div[1]/div[1]/div/div[2]')
		imgList = driver.find_elements_by_class_name('MessageFriendEntry_photo_Dm36uk')
		nameList = driver.find_elements_by_class_name('MessageFriendEntry_name_3XMO59')
		msgList = driver.find_elements_by_class_name('MessageFriendEntry_excerpt_dgiXLz')
		timeList = driver.find_elements_by_class_name('MessageFriendEntry_date_lZl2hR')
		mail_text += '<h3><p>' + msgInfo.text + '</h3>'

		if (len(imgList) == 0):
			logger.info(u'沒有未回覆信件')
			mail_text += '<h4><p>' + u'沒有未回覆信件' + '</h4></p>'
		else:
			for i in range(len(nameList)):
				receive_img = str(imgList[i].get_attribute('style')[23:-3])
				# logger.info(receive_img)
				mail_text += '<h4><img src="' + receive_img
				mail_text += '" width="70px" height="90px"></a>'
				mail_text += '<font size="3" color="#ff1aff">' + nameList[i].text + '<br>'
				mail_text += '&emsp;&emsp;&emsp;&emsp;' + '(' + timeList[i].text + ')' + '</font>'
				mail_text += '<h4><p>' + msgList[i].text + '</h4></p>'

		mail_text += '</body></html>'

		time.sleep(3)
		driver.close()
		return True, mail_text
	except Exception as e:
		driver.close()
		traceback = sys.exc_info()[2]
		logger.error(sys.exc_info())  #
		logger.error(traceback.tb_lineno)  #
		logger.error(e)
		return False, None


def senfMail(alis, text):
	try:
		logger.info(u'發送email')  # 建立mail (發送圖片)
		sender = DCARD_INFO['SMTP_ACCOUNT']  # 'afly.bsky@gmail.com'
		passwd = DCARD_INFO['SMPT_PASSWORD']  # 'hwmaianxun'
		receivers = ['afly.bsky@yahoo.com.tw']
		emails = [elem.strip().split(',') for elem in receivers]
		msg = MIMEMultipart()
		msg['Subject'] = alis + str(today)
		msg['From'] = sender
		msg['To'] = ','.join(receivers)

		msg.preamble = 'Multipart massage.\n'
		part = MIMEText(text, 'html', 'utf-8')
		msg.attach(part)

		smtp = smtplib.SMTP("smtp.gmail.com:587")
		smtp.ehlo()
		smtp.starttls()
		smtp.login(sender, passwd)

		smtp.sendmail(msg['From'], emails, msg.as_string())
		logger.info('Send mails to ' + msg['To'])
		logger.info(u'寄信成功')
	except smtplib.SMTPException:
		logger.error(u'寄信失敗')

	except Exception as e:
		traceback = sys.exc_info()[2]
		logger.error(sys.exc_info())
		logger.error(traceback.tb_lineno)
		logger.error(e)
	finally:
		smtp.quit()


# 網頁成功開啟後開啟logger
try:
	today, yesterday = getDay()
	loggerName = "DCARD"
	path = str(os.path.abspath('.'))
	loggerPath = path + '/DCARD_LOG/DCARD - log - ' + str(today) + '.txt'
	logger = getLogger(loggerName, loggerPath)

	DCARD_INFO = {}
	login_file = 'DCARD_INFO.txt'
	logger.info(u'檢查登入文件:{}'.format(os.path.exists(login_file)))
	if os.path.exists(login_file):
		logger.info(os.path.abspath(login_file))
		with open(login_file, 'r') as f:
			data = f.readlines()
			for d in data:
				d = d.split(' ')
				DCARD_INFO[d[0]] = d[1].replace('\n', '')
		while (True):
			status, mail_text = open_web(DCARD_INFO['NKFUST_ACCOUNT'], DCARD_INFO['NKFUST_PASSWORD'], u'高科狄卡')
			if status != False:
				senfMail(u'高科狄卡', mail_text)
				break
		while (True):
			status, mail_text = open_web(DCARD_INFO['NCKU_ACCOUNT'], DCARD_INFO['NCKU_PASSWORD'], u'成大狄卡')
			if status != False:
				senfMail(u'成大狄卡', mail_text)
				break
		while (True):
			status, mail_text = open_web(DCARD_INFO['NTU_ACCOUNT'], DCARD_INFO['NTU_PASSWORD'], u'台大狄卡')
			if status != False:
				senfMail(u'台大狄卡', mail_text)
				break
	else:
		logger.info(u'找不到登入文件:{}'.format(login_file))

except Exception as e:
	traceback = sys.exc_info()[2]
	print(sys.exc_info())
	print(traceback.tb_lineno)
	print(e)
finally:
	logger.info(u'logger已關閉')
	handlers = logger.handlers[:]
	for handler in handlers:
		handler.close()
		logger.removeHandler(handler)
