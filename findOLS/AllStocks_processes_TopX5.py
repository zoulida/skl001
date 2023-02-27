__author__ = 'zoulida'
#coding: utf-8


from multiprocessing import Pool, Manager
import time
import random
from findOLS import dbQueryTools
import pandas as pd
import tools.tusharePro as tp
ts = tp.getPro()
#import tushare as ts
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import pprint
import statsmodels.formula.api as smf
import statsmodels.tsa.stattools as tsst
import traceback
import os
dict = None
def plot_price_series(df, ts1, ts2):
    #indf = df.index._data
    #print(indf)
    #print(df.index.values[0])
    #print(df.index.values[1])
    #tt = df.index.values[0]
    #print(tt)
    startDatatime = df.index.values[0]
    endDatatime = df.index.values[df.shape[0]-1]
    #print(startDatatime)
    months = mdates.MonthLocator()  # every month
    fig, ax = plt.subplots()
    ax.plot(df.index, df[ts1], label=ts1)
    ax.plot(df.index, df[ts2], label=ts2)
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y')) #%Y-%m-%d
    #ax.set_xlim(datetime.datetime(2012, 1, 1), datetime.datetime(2013, 1, 1))
    ax.set_xlim(startDatatime, endDatatime)
    #ax.set_xlim('2017-12-09', '2018-12-09')
    ax.grid(True)
    fig.autofmt_xdate()

    plt.xlabel('Month/Year')
    plt.ylabel('Price ($)')
    plt.title('%s and %s Daily Prices' % (ts1, ts2))
    plt.legend()
    plt.show()

def plot_price_series_stock(df, ts1, ts2):

    startDatatime = df.index.values[0]
    endDatatime = df.index.values[df.shape[0]-1]
    #print(startDatatime)
    months = mdates.MonthLocator()  # every month
    fig, ax = plt.subplots()
    ax.plot(df.index, df['X'], label=ts1)
    ax.plot(df.index, df['Y'], label=ts2)
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y')) #%Y-%m-%d
    #ax.set_xlim(datetime.datetime(2012, 1, 1), datetime.datetime(2013, 1, 1))
    ax.set_xlim(startDatatime, endDatatime)
    #ax.set_xlim('2017-12-09', '2018-12-09')
    ax.grid(True)
    fig.autofmt_xdate()

    plt.xlabel('Month/Year')
    plt.ylabel('Price ($)')
    #print('dfsdfdsaf ddddddddd',list[list.code == ts1].iloc[0, 2])
    if dict.get('stockRange') == 'hs300':
        namestock1 = list[list.code == ts1].iloc[0, 2]
        namestock2 = list[list.code == ts2].iloc[0, 2]
    else:
        namestock1 = list[list.symbol == ts1].iloc[0,2]#df取值
        namestock2 = list[list.symbol == ts2].iloc[0,2]
        #namestock2 = list.loc[ts2].iloc[2]
        #print('tttttttttt' , namestock1)
    plt.title('%s and %s Daily Prices' % (namestock1, namestock2))
    plt.legend()
    plt.show()
    #print(list)
def cadf(df):
    # Calculate optimal hedge ratio "beta"
    res = smf.ols(formula='Y ~ X ', data=df).fit()
    # res = smf.ols(y=df['WLL'], x=df["AREX"])
    #print(res.summary())
    beta_hr = res.params.X

    beta_hr * df["X"]
    # Calculate the residuals of the linear combination
    df["res"] = df["Y"] - beta_hr * df["X"]

    # Plot the residuals
    #plot_residuals(df)
    #print(df)
    #print(df["res"])
    # Calculate and output the CADF test on the residuals
    cadf = tsst.adfuller(df["res"])
    #print(cadf[1])
    #pprint.pprint(cadf)
    return cadf

def plot_residuals(df):
    months = mdates.MonthLocator()  # every month
    fig, ax = plt.subplots()
    ax.plot(df.index, df["res"], label="Residuals")
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    startDatatime = df.index.values[0]
    endDatatime = df.index.values[df.shape[0]-1]
    ax.set_xlim(startDatatime, endDatatime)
    ax.grid(True)
    fig.autofmt_xdate()

    plt.xlabel('Month/Year')
    plt.ylabel('Price ($)')
    plt.title('Residual Plot')
    plt.legend()

    plt.plot(df["res"])
    plt.show()
def getPAIRdata_memory(code1, code2, dict):#数据整理,基于内存
    #global startdate
    #print(dict.get(code1))
    #print(dict)
    startdate = dict.get('startdate')
    enddate = dict.get('enddate')
    if dict.get(code1) is None:
        dict[code1] = dbQueryTools.queryMySQL(code1, startdate , enddate )


    if dict.get(code2) is None:
        dict[code2] = dbQueryTools.queryMySQL(code2, startdate , enddate )

    df_X = dict.get(code1)
    df_Y = dict.get(code2)
    if df_X.shape[0] < 1:
        return df_X
    df = pd.DataFrame(index=df_Y.index)
    if df_Y.shape[0] < 1:
        return df_Y
    df["X"] = df_X["close"]
    df["Y"] = df_Y["close"]

    df3 = df.sort_index(axis=0, ascending=True)
    #pd.set_option('display.max_rows', None)  # 打印所有行
    df3 = df3.dropna(axis=0, how='any')  # 删除表中任何含有NaN的行
    #print(df3)
    return df3
    #return
def getPAIRdata_database(code1, code2):#数据整理,基于数据库

    df_X = dbQueryTools.queryMySQL(code1, dict.get('startdate'), dict.get('enddate')  )
    df_Y = dbQueryTools.queryMySQL(code2, dict.get('startdate'), dict.get('enddate'))
    #print(df_Y)

    df_X['date1'] = df_X.index
    df_X['date2'] = df_X['date1'] #df_X['date1'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))  # 必须两个datetime
    df_X = df_X.set_index(['date2'])
    # print(df_X)

    df_Y['date1'] = df_Y.index
    df_Y['date2'] = df_Y['date1'] #df_Y['date1'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))  # 必须两个datetime
    df_Y = df_Y.set_index(['date2'])
    # print(df_Y)

    df = pd.DataFrame(index=df_Y.index)
    df["X"] = df_X["close"]
    df["Y"] = df_Y["close"]

    df3 = df.sort_index(axis=0, ascending=True)
    pd.set_option('display.max_rows', None)  # 打印所有行
    df3 = df3.dropna(axis=0, how='any')  # 删除表中任何含有NaN的行
    #print(df3)
    return df3
def getPAIRdata(code1, code2):#数据整理

    df_X = dbQueryTools.queryMySQL(code1, dict.get('startdate'), dict.get('enddate') )
    df_Y = dbQueryTools.queryMySQL(code2, dict.get('startdate'), dict.get('enddate') )
    # df = ts.get_hist_data('600848') #一次性获取全部日k线数据
    '''from findOLS import dbQueryTools
    df_X = dbQueryTools.queryMySQL(code1, startdate , enddate )
    df_Y = dbQueryTools.queryMySQL(code2, startdate , enddate )'''
    #print(df_Y)

    df_X['date1'] = df_X.index
    #df_X['date2'] = df_X['date1'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))  # 必须两个datetime
    #df_X = df_X.set_index(['date2'])
    # print(df_X)

    df_Y['date1'] = df_Y.index
    #df_Y['date2'] = df_Y['date1'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))  # 必须两个datetime
    #df_Y = df_Y.set_index(['date2'])
    # print(df_Y)

    df = pd.DataFrame(index=df_Y.index)
    df["X"] = df_X["close"]
    df["Y"] = df_Y["close"]

    df3 = df.sort_index(axis=0, ascending=True)
    pd.set_option('display.max_rows', None)  # 打印所有行
    df3 = df3.dropna(axis=0, how='any')  # 删除表中任何含有NaN的行
    #print(df3)
    return df3

def compare(code1, code2, dict):#cadf处理
    #df3 = getPAIRdata(code1, code2)
    #df3 = getPAIRdata_database(code1, code2)
    try:
        df3 = getPAIRdata_memory(code1, code2, dict)
        if df3.shape[0] == 0:
            return None
        # Plot the two time series
        #plot_price_series(df3, "X", "Y")

        # Display a scatter plot of the two time series
        # cadf2.plot_scatter_series(df3, "X", "Y")

        start = datetime.datetime.now()
        if df3.shape[0] < 150: #行数小于200以下不给予比较
            return None
        # OLS 回归
        cadfTuple = cadf(df3)
        end = datetime.datetime.now()
        #print('消耗时间 ' , (end - start).total_seconds())
    except Exception as e:
        import traceback
        #print('traceback.print_exc():', traceback.print_exc())
        return None


    return cadfTuple[0]

def showTop10(int,df_result2):
    #df_result2.iloc[1, 0]
    for i in range(int):
        print(df_result2.iloc[i, 0])
        print(df_result2.iloc[i, 1])
        print(df_result2.iloc[i, 2])
        df = getPAIRdata(df_result2.iloc[i, 0], df_result2.iloc[i, 2])
        plot_price_series_stock(df, df_result2.iloc[i, 0], df_result2.iloc[i, 2])

    #print(df_result2.iloc[1, 3])

def hs300gogo(list):
    list2 = list
    lenth = list.__len__()
    # compare('600016','000540')
    df_result = pd.DataFrame(columns=['A.code', 'A.name','B.code', 'B.name', 'adf'])
    i=0
    j = 0
    k = 0

    for index, row in list.iterrows():
        # print(index, row)
        code1 = row['成分券代码']
        name1 = row['成分券名称']
        for index2, row2 in list2.iterrows():
            if i >= dict.get('loopnum'):
                break
            # print (index)
            code2 = row2['成分券代码']
            name2 = row2['成分券名称']
            print('开始测试平稳性   ', name1, ' ', name2 , '  第%s循环的第%s个，总%s个，共%s个循环，最大比较次数为%s'%(j,k,i,lenth,dict.get('loopnum')))
            ss = compare(code1, code2, dict)
            insertRow = pd.DataFrame([[code1, name1, code2, name2, ss]], columns=['A.code', 'A.name','B.code', 'B.name', 'adf'])
            df_result = df_result.append(insertRow, ignore_index=False)
            i = i + 1
            k = k + 1
        j = j+1
        k = 0
    return df_result

def ALLStocksgogo(list):#循环比较
    list2 = list
    # compare('600016','000540')
    df_result = pd.DataFrame(columns=['A', 'B', 'adf'])
    i = 0

    for index, row in list.iterrows():
        # print(index, row)
        code1 = row['symbol']
        for index2, row2 in list2.iterrows():
            if i >= dict.get('loopnum'):
                break
            i = i + 1
            # print (index)

            code2 =  row2['symbol']
            print('开始测试平稳性   ', code1, ' ', code2)
            #ss = compare(index, index2)
            try:
                ss = compare(code1, code2, dict)
            except Exception as e:
                print('traceback.print_exc():', e)
                traceback.print_exc()
                # 如果以上插入过程出错，跳过这条数据记录，继续往下进行
                continue  # break'''
            insertRow = pd.DataFrame([[code1, code2, ss]], columns=['A', 'B', 'adf'])
            df_result = df_result.append(insertRow, ignore_index=False)

    return df_result

def ALLStocksPools(list):#循环比较多线程
    from concurrent.futures import ThreadPoolExecutor,wait,as_completed
    executor = ThreadPoolExecutor(dict.get('max_workers'))
    f_list = []
    list2 = list
    # compare('600016','000540')
    #df_result = pd.DataFrame(columns=['A', 'B', 'adf'])
    i = 0 #计数器

    for index, row in list.iterrows():
        # print(index, row)
        for index2, row2 in list2.iterrows():
            if i >= dict.get('loopnum'):
                break
            i = i + 1
            # print (index)
            print('提交任务   ', index, ' ', index2)

            future = executor.submit(compareTask, index, index2, i)
            f_list.append(future)
    print(wait(f_list))
    return df_result


def poolpre():
    from multiprocessing import cpu_count
    #print(cpu_count())
    pool = Pool(cpu_count()+2)
    return pool
def saveMiddle(pool, res_l):
    print("保存中间结果~~~~~~~~~~~~~~~~~~~~~~")
    pool.close()
    pool.join()   #调用join之前，先调用close函数，否则会出错。执行完close后不会有新的进程加入到pool,join函数等待所有子进程结束
    addList(res_l)

def addList(res_l):
    listtemp5 = []
    global listTop
    for res in res_l:
        #spend = (timest - datetime.datetime.now()).total_seconds()
        #time = datetime.datetime.now()
        #print('正在提取结果：  ', listtemp5.__len__(),'   ; 上次消耗时间 ' , spend)
        try:
            #item = res.get()
            #adfitem = item[2]
            adfitem = res[4]

            import math
            if adfitem is None or math.isnan(adfitem) :
                continue
            listtemp5.append(res)
            #df_result = df_result.append(res.get(), ignore_index=False) #df_result.append(res.get())
        except Exception as e:
            print('traceback.print_exc():', e)
            traceback.print_exc()
    if listTop is not None:
        listtemp5.extend(listTop)
    print('list5= ' , listtemp5)
    #import time
    #time.sleep(100) # 休眠10秒
    import heapq
    listTop = heapq.nlargest(500, listtemp5, key=lambda x: -x[4])
    return

def ALLStocksProcessPools(list):#循环比较多进程
    res_l=[]
    start = time.time()
    pool = poolpre()
    #from concurrent.futures import ThreadPoolExecutor,wait,as_completed
    #executor = ThreadPoolExecutor(max_workers=3)
    list2 = list
    maxMemoryloop = 10000 #中间结果保存内存数量
    #list2 = list
    lenth = list.__len__()
    # compare('600016','000540')
    #df_result = pd.DataFrame(columns=['A.code', 'A.name','B.code', 'B.name', 'adf'])
    i=0
    j = 0
    k = 0

    for index, row in list.iterrows():
        code1 = row['symbol']
        name1 = row['name']
        loopnum = dict.get('loopnum')
        if i >= loopnum:
            break
        for index2, row2 in list2.iterrows():
            code2 = row2['symbol']
            name2 = row2['name']
            if code1 == code2:
                break
            if i % maxMemoryloop == 0 and i != 0:
                saveMiddle(pool,res_l)
                res_l=[]
                pool = poolpre()
            i = i + 1
            # print (index)
            print('开始测试平稳性   ', name1, ' ', name2 , '  第%s循环的第%s个，总%s个，共%s个循环，最大比较次数为%s'%(j,k,i,lenth,dict.get('loopnum')))
            appresult = pool.apply_async(compareTask, (code1, code2, i, dict), callback=None)#apply_async(func[, args[, kwds[, callback[, error_callback]]]])
            ss = appresult.get()
            insertRow = (code1, name1, code2, name2, ss)#, columns=['A.code', 'A.name','B.code', 'B.name', 'adf'])

            print('任务队列大小: ', pool.__sizeof__())
            #print('任务队列大小: ', pool._taskqueue._qsize())
            if pool.__sizeof__() > 10000:#防止任务队列内存占用过大
                time.sleep(10) # 休眠10秒
            #import sys as sys
            #print ('对象大小 ', sys.getsizeof(pool._cache))
            res_l.append(insertRow)
            #print(insertRow.get())
            #df_result = df_result.append(insertRow, ignore_index=False)
            i = i + 1
            k = k + 1
        j = j+1
        k = 0
    saveMiddle(pool,res_l)
    print ("Sub-process(es) done.")
    stop = time.time()
    print ('delay: %.3fs' % (stop - start))
    #print(df_result)
    return res_l



def compareTask(index, index2, i, dict):
    print('开始测试平稳性   ', index, ' ', index2, ' ', i)
    print ("\nRun task 进程ID-%s" %(os.getpid()) )#os.getpid()获取当前的进程的ID)
    try:
        ss = compare(index, index2, dict)
        if ss is None:
            return None
        #insertRow = pd.DataFrame([[index, index2, ss]], columns=['A', 'B', 'adf'])
        #insertRow = (index, index2, ss )
        #global df_result #如果确定要引用全局变量，并且要对它修改，必须加上global关键字。
        #df_result = df_result.append(insertRow, ignore_index=False)
    except Exception as e:
        print('traceback.print_exc():', e)
        traceback.print_exc()
        # 如果以上插入过程出错，跳过这条数据记录，继续往下进行
        #continue  # break'''
    return ss

def mycallback(insertRow):#多线程调用
    start1 = datetime.datetime.now()
    print( str(insertRow))
    #with open('muti_process_log.txt', 'a+') as f:
    #    f.writelines('%d, \n' % x)
    global df_result #如果确定要引用全局变量，并且要对它修改，必须加上global关键字。
    df_result = df_result.append(insertRow, ignore_index=False)#怀疑数据量大时候非常的耗时
    end1 = datetime.datetime.now()
    print('df_result.append 消耗时间 ' , (end1 - start1).total_seconds())

def sayHi(num):#作废
    #print('num ', num)
    #time.sleep(random.random()) #random.random()随机生成0-1之间的小数
    try:
        insertRow = pd.DataFrame([['dd', 'cc', num]], columns=['A', 'B', 'adf'])
        #global df_result #如果确定要引用全局变量，并且要对它修改，必须加上global关键字。
        #df_result = df_result.append(insertRow, ignore_index=False)
    except Exception as e:
        print('traceback.print_exc():', e)
        traceback.print_exc()

    #print(num)
    return insertRow



listTop = None

def flow():
    plt.rcParams['font.family'] = 'SimHei' #解决plt中文乱码


    #iii = dict.get('loopnum')

    #ss = compare('001222', '000001', dict)


    start = datetime.datetime.now()


    if stockRange == 'hs300':
        import akshare
        hs300 = akshare.index_stock_cons_weight_csindex(symbol="000300")#获取沪深300成分股列表
        #list = ts.index_weight(index_code='399300.SZ', start_date='20180901', end_date='20180930')
        df_result = hs300gogo(hs300)
    if stockRange == 'all':
        list = ts.stock_basic()
        df_result = ALLStocksgogo(list)
    if stockRange == 'allPools':
        max_workers = 10
        dict['max_workers'] = max_workers
        list = ts.get_stock_basics()
        df_result = ALLStocksPools(list)
    if stockRange == 'ALLStocksProcessPools':
        from multiprocessing import cpu_count
        max_workers = cpu_count() * 2 + 2
        dict['max_workers'] = max_workers
        list = ts.stock_basic()
        list = list[list['market'] == '主板']
        startPools = datetime.datetime.now()
        ALLStocksProcessPools(list)
        endPools = datetime.datetime.now()
        #print('多进程消耗时间 ' , (endPools - startPools).total_seconds())
        #timest = datetime.datetime.now()

        df_result = pd.DataFrame(listTop, columns=['A.code', 'A.name','B.code', 'B.name', 'adf'])
        #df_result = df_result.append(listtemp5)
    end = datetime.datetime.now()
    #print('消耗时间 ' , (end - start).total_seconds())
    #df_result = df_result.dropna()#去除NAN
    #df_result2 = df_result.sort_values(by = 'adf',axis = 0,ascending = True )#排序adf

    print(df_result)
    #print(df_result2.dropna() )
    print('消耗时间 ' , (end - start).total_seconds())
    #print('多进程消耗时间 ' , (endPools - startPools).total_seconds())
    df_result.to_csv('%s_%s_%s_%s_Result.csv'% (end.year, end.month, end.day, end.timestamp()))


    '''import sys as sys
    
    a = [x for x in range(1000)]
    print ('对象大小 ', sys.getsizeof(df_result2))'''

    #bb=pd.DataFrame(df_result2)
    #print(bb)
    showTop10(5,df_result)

if __name__ == '__main__':
    m = Manager()
    dict = m.dict()
    #print(dict)
    df_result = pd.DataFrame()#pd.DataFrame(columns=['A', 'B', 'adf'])

    startdate = '2015-12-09'
    enddate = '2018-12-23'
    loopnum = 1000000000000 #最大比较次数        # 100000约需要20分钟
    stockRange = 'ALLStocksProcessPools' # 用多线程完成
    dict['startdate'] = startdate
    dict['enddate'] = enddate
    dict['loopnum'] = loopnum
    dict['stockRange'] = stockRange
    flow()



