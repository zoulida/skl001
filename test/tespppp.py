__author__ = 'zoulida'
#coding: utf-8


from multiprocessing import Pool
import time
import random
from findOLS import dbQueryTools
import pandas as pd
import tushare as ts
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import pprint
import statsmodels.formula.api as smf
import statsmodels.tsa.stattools as tsst
import traceback

def mycallback(insertRow):#多线程调用
    print( str(insertRow))
    #with open('muti_process_log.txt', 'a+') as f:
    #    f.writelines('%d, \n' % x)
    global df_result #如果确定要引用全局变量，并且要对它修改，必须加上global关键字。
    df_result = df_result.append(insertRow, ignore_index=False)
def sayHi(num):
    #print('num ', num)
    time.sleep(random.random()) #random.random()随机生成0-1之间的小数
    try:
        insertRow = pd.DataFrame([['dd', 'cc', num]], columns=['A', 'B', 'adf'])
        #global df_result #如果确定要引用全局变量，并且要对它修改，必须加上global关键字。
        #df_result = df_result.append(insertRow, ignore_index=False)
    except Exception as e:
        print('traceback.print_exc():', e)
        traceback.print_exc()

    #print(num)
    return insertRow

if __name__ == '__main__':
    df_result = pd.DataFrame(columns=['A', 'B', 'adf'])
    start = time.time()
    pool = Pool(10)
    for i in range(100):
        pool.apply_async(sayHi, (i,), callback=mycallback)
    pool.close()
    pool.join()
    stop = time.time()
    print ('delay: %.3fs' % (stop - start))
    print(df_result)