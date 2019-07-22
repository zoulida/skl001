__author__ = 'zoulida'
#!/usr/bin/env python
# -*- coding:utf-8 -*-

from pymongo import MongoClient

conn = MongoClient('202.194.246.155', 27017)
db = conn.mydb  #连接mydb数据库，没有则自动创建

#from tools.mongodbFactory import getConnectionWuDuJi
#wenduji = getConnectionWuDuJi()
wenduji = db.WuDuJi

my_set = wenduji #使用test_set集合，没有则自动创建

my_set.insert({"name":"zhangsan","age":18})
#或
#my_set.save({"name":"zhangsan","age":18})

#添加多条数据到集合中
users=[{"name":"zhangsan","age":18},{"name":"lisi","age":20}]
#my_set.insert(users)
#或
#my_set.save(users)

#查询全部
# for i in my_set.find():
#     print(i)
#查询name=zhangsan的
for i in my_set.find({"name":"zhangsan"}):
    print(i)
for i in my_set.find():
    print(i)
#print(my_set.find_one({"name":"zhangsan"}))