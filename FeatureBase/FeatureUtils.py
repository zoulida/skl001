__author__ = 'zoulida'

################ Bollinger Bands #############################

# Load the necessary packages and modules
import pandas as pd
import matplotlib.pyplot as plt
import talib
import datetime


# Compute the Bollinger Bands
def BBANDS(data, ndays = 12):

    #MA = pd.Series(pd.rolling_mean(data['close'], ndays))
    MA = pd.Series(data['Close']).rolling(window=ndays).mean()
    SD = pd.Series(data['Close']).rolling(window=ndays,center=False).std()
    #SD = pd.Series(pd.rolling_std(data['Close'], ndays))

    b1 = MA + (2 * SD)
    B1 = pd.Series(b1, name = 'Upper BollingerBand')
    data = data.join(B1)
    #print(data)

    b2 = MA - (2 * SD)
    B2 = pd.Series(b2, name = 'Lower BollingerBand')
    data = data.join(B2)

    return data

def CCI(data, ndays=20):
    TP = (data['High'] + data['Low'] + data['Close']) / 3
    CCI = pd.Series((TP - pd.rolling_mean(TP, ndays)) / (0.015 * pd.rolling_std(TP, ndays)),
    name = 'CCI')
    data = data.join(CCI)
    return data

# Ease of Movement
def EVM(data, ndays=14):
    dm = ((data['High'] + data['Low'])/2) - ((data['High'].shift(1) + data['Low'].shift(1))/2)
    br = (data['Volume'] / 100000000) / ((data['High'] - data['Low']))
    EVM = dm / br
    EVM_MA = pd.Series(pd.rolling_mean(EVM, ndays), name = 'EVM')
    data = data.join(EVM_MA)
    return data

# Force Index
def ForceIndex(data, ndays=1):
    FI = pd.Series(data['Close'].diff(ndays) * data['Volume'], name = 'ForceIndex')
    data = data.join(FI)
    return data

# Simple Moving Average
def SMA(data, ndays=12):
    SMA = pd.Series(pd.rolling_mean(data['Close'], ndays), name = 'SMA')
    data = data.join(SMA)
    return data

# Simple Moving Average of Volume
def ADV(data, ndays=12):
    ADV = pd.Series(pd.rolling_mean(data['成交金额'], ndays), name = 'ADV')
    data = data.join(ADV)
    return data

# Exponentially-weighted Moving Average
def EWMA(data, ndays=12):
    EMA = pd.Series(pd.ewma(data['Close'], span = ndays, min_periods = ndays - 1),
    name = 'EWMA_' + str(ndays))
    data = data.join(EMA)
    return data

# Rate of Change (ROC)
def ROC(data,n=5):
    N = data['Close'].diff(n)
    D = data['Close'].shift(n)
    ROC = pd.Series(N/D,name='Rate of Change')
    data = data.join(ROC)
    return data

def OBV(data):
    real = talib.OBV(data.Close, data.Volume)
    OBV = pd.Series(real, name = 'OBV')
    data = data.join(OBV)
    return data

if __name__ == "__main__":
    '''#FeatureUtils.py
    start = datetime.datetime(2012, 1, 1)
    end = datetime.datetime(2013, 1, 1)
    NSEI = web.DataReader("AREX", "yahoo", start, end)
    data = pd.DataFrame(NSEI)
    #print(data)
    # Compute the Bollinger Bands for NIFTY using the 50-day Moving average
    n = 12
    NIFTY_BBANDS = BBANDS(data, n)
    print(NIFTY_BBANDS)'''

    startdate  = '2017-12-09'
    enddate  = '2018-12-13'
    code = '600016'
    import DBStock.dbQueryPools as dbpool
    data = dbpool.queryMySQL_plot_stock_market(code, startdate, enddate)
    #print(data)


    '''n = 12
    NIFTY_BBANDS = BBANDS(data, n)
    print(NIFTY_BBANDS)'''

    '''n = 20
    NIFTY_CCI = CCI(data, n)
    CCI = NIFTY_CCI['CCI']
    print(NIFTY_CCI)

    # Plotting the Price Series chart and the Commodity Channel index below
    fig = plt.figure(figsize=(7,5))
    ax = fig.add_subplot(2, 1, 1)
    ax.set_xticklabels([])
    plt.plot(data['Close'],lw=1)
    plt.title('NSE Price Chart')
    plt.ylabel('Close Price')
    plt.grid(True)
    bx = fig.add_subplot(2, 1, 2)
    plt.plot(CCI,'k',lw=0.75,linestyle='-',label='CCI')
    plt.legend(loc=2,prop={'size':9.5})
    plt.ylabel('CCI values')
    plt.grid(True)
    plt.setp(plt.gca().get_xticklabels(), rotation=30)
    plt.show()'''

    '''# Compute the 14-day Ease of Movement for AAPL
    n = 14
    AAPL_EVM = EVM(data, n)
    EVM = AAPL_EVM['EVM']

    # Plotting the Price Series chart and the Ease Of Movement below
    fig = plt.figure(figsize=(7,5))
    ax = fig.add_subplot(2, 1, 1)
    ax.set_xticklabels([])
    plt.plot(data['Close'],lw=1)
    plt.title('AAPL Price Chart')
    plt.ylabel('Close Price')
    plt.grid(True)
    bx = fig.add_subplot(2, 1, 2)
    plt.plot(EVM,'k',lw=0.75,linestyle='-',label='EVM(14)')
    plt.legend(loc=2,prop={'size':9})
    plt.ylabel('EVM values')
    plt.grid(True)
    plt.setp(plt.gca().get_xticklabels(), rotation=30)
    plt.show()'''

    '''# Compute the Force Index for Apple
    n = 1
    AAPL_ForceIndex = ForceIndex(data,n)
    print(AAPL_ForceIndex)'''

    '''# Compute the 50-day SMA for NIFTY
    n = 50
    SMA_NIFTY = SMA(data,n)
    SMA_NIFTY = SMA_NIFTY.dropna()
    SMA = SMA_NIFTY['SMA']

    # Compute the 200-day EWMA for NIFTY
    ew = 200
    EWMA_NIFTY = EWMA(data,ew)
    EWMA_NIFTY = EWMA_NIFTY.dropna()
    EWMA = EWMA_NIFTY['EWMA_200']

    # Plotting the NIFTY Price Series chart and Moving Averages below
    plt.figure(figsize=(9,5))
    plt.plot(data['Close'],lw=1, label='NSE Prices')
    plt.plot(SMA,'g',lw=1, label='50-day SMA (green)')
    plt.plot(EWMA,'r', lw=1, label='200-day EWMA (red)')
    plt.legend(loc=2,prop={'size':11})
    plt.grid(True)
    plt.setp(plt.gca().get_xticklabels(), rotation=30)
    plt.show()'''

    '''# Compute the 5-period Rate of Change for NIFTY
    n = 5
    NIFTY_ROC = ROC(data,n)
    ROC = NIFTY_ROC['Rate of Change']

    # Plotting the Price Series chart and the Ease Of Movement below
    fig = plt.figure(figsize=(7,5))
    ax = fig.add_subplot(2, 1, 1)
    ax.set_xticklabels([])
    plt.plot(data['Close'],lw=1)
    plt.title('NSE Price Chart')
    plt.ylabel('Close Price')
    plt.grid(True)
    bx = fig.add_subplot(2, 1, 2)
    plt.plot(ROC,'k',lw=0.75,linestyle='-',label='ROC')
    plt.legend(loc=2,prop={'size':9})
    plt.ylabel('ROC values')
    plt.grid(True)
    plt.setp(plt.gca().get_xticklabels(), rotation=30)
    plt.show()'''

    dataResult = OBV(data)
    print(dataResult)