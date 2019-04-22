import logging
import os

import sys
import time
import datetime as dt
from datetime import datetime

class Log:
	def __init__(self,loggerName):
		today, yesterday = self.getDay()
		# loggerName = "NEWS"
		path = str(os.path.abspath('.'))
		loggerPath = path + '/{}_LOG/log - {}.txt'.format(loggerName,str(today))
		self.getLogger(loggerName, loggerPath)

	def getDay(self):
		now = datetime.now()
		today = str(now.strftime("%Y-%m-%d"))

		now -= dt.timedelta(days=1)
		yesterday = "{}月{}日".format(now.month, now.day)
		return today, yesterday

	def getLogger(self,loggerName, loggerPath):
		# 設置logger
		self.logger = logging.getLogger(loggerName)  # 不加名稱設置root logger
		self.logger.setLevel(logging.DEBUG)
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
		self.logger.addHandler(ch)
		self.logger.addHandler(fh)
		# Handler只啟動一次
		# 設置logger
		self.logger.info(u'logger已啟動')
	

	def closeLog(self):
		self.logger.info(u'logger已關閉')
		handlers = self.logger.handlers[:]
		for handler in handlers:
			handler.close()
			self.logger.removeHandler(handler)

