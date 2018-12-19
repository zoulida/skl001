__author__ = 'Administrator'
import pandas as pd
import pymysql
import datetime


# 数据库名称和密码
name = 'root'
password = 'root'  # 替换为自己的账户名和密码


def queryMySQL(code1, startdate = '2017-12-09', enddate = '2018-12-09'):
    ###########################查询刚才操作的成果##################################

    # 重新建立数据库连接
    db = pymysql.connect('localhost', name, password, 'stockDataBase')
    cursor = db.cursor()
    # 查询数据库并打印内容
    #cursor.execute('select * from stock_600016 where timestamp = 977155200')
    #str = 'select * from stock_600016 where 日期 between %s' % startdata + 'and %s' % enddata
    #print(str)
    code = 'stock_'+code1
    #print(code)
    cursor.execute('select * from %s where 日期 between \'%s\'' % (code , startdate) + ' and \'%s\'' % enddate)
    #cursor.execute('select * from stock_600016 where 收盘价 = 18.56')
    results = cursor.fetchall()
    df = pd.DataFrame(list(results))
    #df2 = df.rename(columns={'0': 'timestamp', '1': 'data', '4': 'close', '5': 'price', '6': 'e'} , inplace=True)
    df.rename(columns={1:'date', 4: 'close'} , inplace=True)
    df2 = df.set_index(['date'])
    #print(df)
    #print(df.columns.values.tolist())
    #print(df2)
    #for row in results:
    #    print(row)
    # 关闭
    cursor.close()
    db.commit()
    db.close()
    return df2

if __name__ == "__main__":
    startdate  = '2017-12-09'
    enddate  = '2018-12-13'
    print(startdate)
    code = '600016'
    queryMySQL(code, startdate, enddate)