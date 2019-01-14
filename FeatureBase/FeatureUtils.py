__author__ = 'zoulida'

################ Bollinger Bands #############################

# Load the necessary packages and modules
import pandas as pd
import datetime
#import pandas.io.data as web
from pandas_datareader import data as web
#import pandas_datareader.data as web

# Compute the Bollinger Bands
def BBANDS(data, ndays = 50):

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
    print(data)
    n = 12
    NIFTY_BBANDS = BBANDS(data, n)
    print(NIFTY_BBANDS)