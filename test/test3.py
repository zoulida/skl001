__author__ = 'Administrator'
import pandas as pd
import pymysql
import datetime

# 数据库名称和密码
name = 'root'
password = 'root'  # 替换为自己的账户名和密码


def queryMySQL():
    ###########################查询刚才操作的成果##################################

    # 重新建立数据库连接
    db = pymysql.connect('localhost', name, password, 'stockDataBase')
    cursor = db.cursor()
    # 查询数据库并打印内容
    #cursor.execute('select * from stock_600016 where timestamp = 977155200')
    #str = 'select * from stock_600016 where 日期 between %s' % startdata + 'and %s' % enddata
    #print(str)
    cursor.execute('select * from stock_600016 where 日期 between \'%s\'' % startdata + ' and \'%s\'' % enddata)
    #cursor.execute('select * from stock_600016 where 收盘价 = 18.56')
    results = cursor.fetchall()
    df = pd.DataFrame(list(results))
    print(df)
    #for row in results:
    #    print(row)
    # 关闭
    cursor.close()
    db.commit()
    db.close()


startdata  = '2017-12-09'
enddata  = '2018-12-13'
print(startdata)
queryMySQL()