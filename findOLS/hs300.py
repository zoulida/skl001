from findOLS import cadf2
import pandas as pd
import tushare as ts
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime

def plot_price_series(df, ts1, ts2):
    months = mdates.MonthLocator()  # every month
    fig, ax = plt.subplots()
    ax.plot(df.index, df[ts1], label=ts1)
    ax.plot(df.index, df[ts2], label=ts2)
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y')) #%Y-%m-%d
    #ax.set_xlim(datetime.datetime(2012, 1, 1), datetime.datetime(2013, 1, 1))
    ax.set_xlim(datetime.datetime(2017, 12, 9), datetime.datetime(2018, 12, 9))
    #ax.set_xlim('2017-12-09', '2018-12-09')
    ax.grid(True)
    fig.autofmt_xdate()

    plt.xlabel('Month/Year')
    plt.ylabel('Price ($)')
    plt.title('%s and %s Daily Prices' % (ts1, ts2))
    plt.legend()
    plt.show()


list = ts.get_hs300s()

#print(list)

df_X = ts.get_hist_data('600038',start='2017-12-09',end='2018-12-09')
df_Y = ts.get_hist_data('600016',start='2017-12-09',end='2018-12-09')
#df = ts.get_hist_data('600848') #一次性获取全部日k线数据
#print(df_Y)

df_X['date1'] = df_X.index
df_X['date2']=df_X['date1'].apply(lambda x:datetime.datetime.strptime(x,'%Y-%m-%d'))#必须两个datetime
df_X = df_X.set_index(['date2'])
#print(df_X)

df_Y['date1'] = df_Y.index
df_Y['date2'] = df_Y['date1'].apply(lambda x:datetime.datetime.strptime(x,'%Y-%m-%d'))#必须两个datetime
df_Y = df_Y.set_index(['date2'])
#print(df_Y)

df = pd.DataFrame(index=df_Y.index)
df["X"] = df_X["close"]
df["Y"] = df_Y["close"]

df3 = df.sort_index(axis=0,ascending=True)
print(df3)

# Plot the two time series
plot_price_series(df3, "X", "Y")

# Display a scatter plot of the two time series
cadf2.plot_scatter_series(df3, "X", "Y")