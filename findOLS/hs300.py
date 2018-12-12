from findOLS import cadf2
import pandas as pd
import tushare as ts
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import pprint
import statsmodels.formula.api as smf
import statsmodels.tsa.stattools as tsst

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
    if stockRange == 'hs300':
        namestock1 = list[list.code == ts1].iloc[0, 2]
        namestock2 = list[list.code == ts2].iloc[0, 2]
    else:
        namestock1 = list.loc[ts1].iloc[0]
        namestock2 = list.loc[ts2].iloc[0]
        #print('tttttttttt' , namestock1)
    plt.title('%s and %s Daily Prices' % (namestock1, namestock2))
    plt.legend()
    plt.show()
    #print(list)
def cadf(df):
    # Calculate optimal hedge ratio "beta"
    res = smf.ols(formula='Y ~ X ', data=df).fit()
    # res = smf.ols(y=df['WLL'], x=df["AREX"])
    print(res.summary())
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
    pprint.pprint(cadf)
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
def getPAIRdata(code1, code2):
    df_X = ts.get_hist_data(code1, start='2017-12-09', end='2018-12-09')
    df_Y = ts.get_hist_data(code2, start='2017-12-09', end='2018-12-09')
    # df = ts.get_hist_data('600848') #一次性获取全部日k线数据
    # print(df_Y)

    df_X['date1'] = df_X.index
    df_X['date2'] = df_X['date1'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))  # 必须两个datetime
    df_X = df_X.set_index(['date2'])
    # print(df_X)

    df_Y['date1'] = df_Y.index
    df_Y['date2'] = df_Y['date1'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))  # 必须两个datetime
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

def compare(code1, code2):
    df3 = getPAIRdata(code1, code2)
    if df3.shape[0] == 0:
        return False
    # Plot the two time series
    #plot_price_series(df3, "X", "Y")

    # Display a scatter plot of the two time series
    # cadf2.plot_scatter_series(df3, "X", "Y")

    start = datetime.datetime.now()
    if df3.shape[0] < 10:
        return None
    # OLS 回归
    cadfTuple = cadf(df3)
    end = datetime.datetime.now()
    #print('消耗时间 ' , (end - start).total_seconds())
    return cadfTuple[0]

def showTop10(int,df_result2):
    #df_result2.iloc[1, 0]
    for i in range(int):
        print(df_result2.iloc[i, 0])
        print(df_result2.iloc[i, 1])
        print(df_result2.iloc[i, 2])
        df = getPAIRdata(df_result2.iloc[i, 0], df_result2.iloc[i, 1])
        plot_price_series_stock(df, df_result2.iloc[i, 0], df_result2.iloc[i, 1])

    #print(df_result2.iloc[1, 3])

def hs300gogo(list):
    list2 = list
    # compare('600016','000540')
    df_result = pd.DataFrame(columns=['A', 'B', 'adf'])
    i = 0

    for index, row in list.iterrows():
        # print(index, row)
        for index2, row2 in list2.iterrows():
            if i >= 10:
                break
            # print (index)
            print('开始测试平稳性   ', row.code, ' ', row2.code)
            ss = compare(row.code, row2.code)
            insertRow = pd.DataFrame([[row.code, row2.code, ss]], columns=['A', 'B', 'adf'])
            df_result = df_result.append(insertRow, ignore_index=False)
            i = i + 1
    return df_result

def ALLStocksgogo(list):
    list2 = list
    # compare('600016','000540')
    df_result = pd.DataFrame(columns=['A', 'B', 'adf'])
    i = 0

    for index, row in list.iterrows():
        # print(index, row)
        for index2, row2 in list2.iterrows():
            if i >= 90000:
                break
            # print (index)
            print('开始测试平稳性   ', index, ' ', index2)
            try:
                ss = compare(index, index2)
            except Exception as e:
                print('traceback.print_exc():', e)
                # 如果以上插入过程出错，跳过这条数据记录，继续往下进行
                continue  # break
            insertRow = pd.DataFrame([[index, index2, ss]], columns=['A', 'B', 'adf'])
            df_result = df_result.append(insertRow, ignore_index=False)
            i = i + 1
    return df_result

#if __name__ == "__main__":#######################################=================================================================
plt.rcParams['font.family'] = 'SimHei' #解决plt中文乱码
#stockRange = 'hs300' # all
stockRange = 'all' # all
start = datetime.datetime.now()
if stockRange == 'hs300':
    list = ts.get_hs300s()
    df_result = hs300gogo(list)
else:
    list = ts.get_stock_basics()
    df_result = ALLStocksgogo(list)
#print('ttttttttttttttt',df_result)

end = datetime.datetime.now()
print('消耗时间 ' , (end - start).total_seconds())
df_result2 = df_result.sort_values(by = 'adf',axis = 0,ascending = True )#排序adf
print(df_result2)
showTop10(5,df_result2)
#print(list)

