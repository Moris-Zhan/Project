from pymongo import MongoClient
class MongoDB:
	def __init__(self):
		host = '140.116.39.114'
		port = 27017
		self.user_name = 'root'
		self.user_pwd = '1234'
		self.user_db = 'admin'
		mechanism = 'SCRAM-SHA-1'

		uri = "mongodb://{username}:{password}@{host}:{port}/{user_db}?authMechanism=SCRAM-SHA-1".format(
			username=self.user_name,
			password=self.user_pwd,
			host=host,
			port=port,
			user_db=self.user_db)

		print("conn_mongo -- uri: " + uri)
		self.mongo_client = MongoClient(uri)

	def conn_db(self,db_name):
		print("Auth : ", self.mongo_client[db_name].authenticate(self.user_name,
																 self.user_pwd,
																 self.user_db,
																 mechanism='SCRAM-SHA-1'))
		mongo_db = self.mongo_client[db_name]
		print("Connect to db : %s " % (db_name))
		self.mongo_db = mongo_db

	def searchInDB(self,key,db_col='exchange'):
		mongo_coll = self.mongo_db[db_col]
		cursor = mongo_coll.find(key)  # 此處須注意，其回傳的並不是資料本身，你必須在迴圈中逐一讀出來的過程中，它才真的會去資料庫把資料撈出來給你。
		return cursor.count()>0

	def insert(self,data,db_col='exchange'):
		try:
			mongo_coll = self.mongo_db[db_col]
			mongo_coll.insert(data)
		except Exception as e:
			print("Insert Error")

	def searchInDBCount(self,key,db_col='exchange'):
		mongo_coll = self.mongo_db[db_col]
		cursor = mongo_coll.find(key)  # 此處須注意，其回傳的並不是資料本身，你必須在迴圈中逐一讀出來的過程中，它才真的會去資料庫把資料撈出來給你。
		print("Search key {} in DB :{} Documents".format(key,cursor.count()))

	def count(self,key,db_col='exchange'):
		mongo_coll = self.mongo_db[db_col]		
		print("Toal {} douments in DB".format(mongo_coll.count()))