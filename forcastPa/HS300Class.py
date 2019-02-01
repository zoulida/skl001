
from __future__ import print_function
__author__ = 'zoulida'
import datetime
import numpy as np
import pandas as pd
import sklearn

#from pandas.io.data import DataReader
#import pandas_datareader.data as web
from pandas_datareader.data import DataReader
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.lda import LDA
from sklearn.metrics import confusion_matrix
from sklearn.qda import QDA
from sklearn.svm import LinearSVC, SVC


def create_lagged_series(symbol, startdate, enddate, lags=5):

    # Obtain stock information from Yahoo Finance
    '''ts = DataReader(
    	symbol, "yahoo",
    	startdate-datetime.timedelta(days=365),
    	enddate
    )'''

    import DBStock.dbQueryPools as dbpool
    ts = dbpool.queryMySQL_plot_stock_market(symbol, startdate, enddate)
    ts['Date'] = pd.to_datetime(ts['Date'])
    ts.set_index('Date', inplace=True)
    ts = ts[(True^ts['Close'].isin([0]))]#条件删除去除值为0的行
    #打印不换行
    #pd.set_option('display.height',1000)
    #pd.set_option('display.max_rows',500)
    #pd.set_option('display.max_columns',500)
    pd.set_option('display.width',1000)
    print(ts)

    # Create the new lagged DataFrame
    tslag = pd.DataFrame(index=ts.index)
    tslag["Today"] = ts["Close"]
    tslag["Volume"] = ts["Volume"]


    # Create the shifted lag series of prior trading period close values
    for i in range(0, lags):
        tslag["Lag%s" % str(i+1)] = ts["Close"].shift(i+1)
    #print(tslag)

    # Create the returns DataFrame
    tsret = pd.DataFrame(index=tslag.index)
    #tsret["Volume"] = tslag["Volume"]
    tsret["Today"] = tslag["Today"].pct_change()*100.0
    #tsret["Close"] = ts["Close"]
    tsret = tsret.join(ts)
    #print('tsret')
    #print(tsret)

    # If any of the values of percentage returns equal zero, set them to
    # a small number (stops issues with QDA model in scikit-learn)
    for i,x in enumerate(tsret["Today"]):
        if (abs(x) < 0.0001):
            tsret["Today"][i] = 0.0001
    #print(tsret)

    # Create the lagged percentage returns columns
    for i in range(0, lags):
        tsret["Lag%s" % str(i+1)] = \
        tsret["Today"].shift(i+1)
        #tslag["Lag%s" % str(i+1)].pct_change()*100.0
        #if (abs(tsret["Lag%s" % str(i+1)]) < 0.0001):
        #    tsret["Lag%s" % str(i+1)] = 0.0001
    #print(tsret)

    # Create the "Direction" column (+1 or -1) indicating an up/down day
    tsret["Direction"] = np.sign(tsret["Today"].shift(-1))
    tsret = tsret[tsret.index >= startdate]

    return tsret

def addFeature(snpret):
    import FeatureBase.FeatureUtils as FU
    snpret = FU.CCI(snpret)
    snpret = FU.BBANDS(snpret)
    snpret = FU.EVM(snpret)
    snpret = FU.ForceIndex(snpret)
    snpret = FU.SMA(snpret)
    snpret = FU.EWMA(snpret)
    snpret = FU.ROC(snpret)
    snpret = FU.OBV(snpret)
    import talib
    real = talib.AD(snpret.High, snpret.Low, snpret.Close, snpret.Volume)
    AD = pd.Series(real, name = 'AD')
    snpret = snpret.join(AD)
    #snpret = FU.BBANDS(snpret)
    #snpret = FU.BBANDS(snpret)
    return snpret

if __name__ == "__main__":
    startdate  = '2011-05-01'
    enddate  = '2018-12-29'
    symbol = '600016'
    # Create a lagged series of the S&P500 US stock market index
    snpret = create_lagged_series(
    	symbol, startdate,
    	enddate, lags=5
    )

    #print(snpret)

    snpret = addFeature(snpret)
    print(snpret)
    snpret = snpret.dropna(axis=0, how='any')  # 删除表中任何含有NaN的行

     # Use the prior two days of returns as predictor
    # values, with direction as the response
    X = snpret[["CCI","Close","换手率","Volume","Today","Lag1","Lag2","Lag3","Lag4","Lag5","Upper BollingerBand",
                "Lower BollingerBand","EVM","ForceIndex","SMA","Rate of Change","OBV","AD"]]
    y = snpret["Direction"]

    # The test data is split into two parts: Before and after 1st Jan 2005.
    start_test = datetime.datetime(2018,1,1)

    # Create training and test sets
    X_train = X[X.index < start_test]
    X_test = X[X.index >= start_test]
    y_train = y[y.index < start_test]
    y_test = y[y.index >= start_test]
    #print(X_test)
    #print(y_test)
    #print(y_test.shape)

    # Create the (parametrised) models
    print("Hit Rates/Confusion Matrices:\n")
    models = [("LR", LogisticRegression()),
              ("LDA", LDA()),
              ("QDA", QDA()),
              ("LSVC", LinearSVC()),
              ("RSVM", SVC(
              	C=1000000.0, cache_size=200, class_weight=None,
                coef0=0.0, degree=3, gamma=0.0001, kernel='rbf',
                max_iter=-1, probability=False, random_state=None,
                shrinking=True, tol=0.001, verbose=False)
              ),
              ("RF", RandomForestClassifier(
              	n_estimators=1000, criterion='gini',
                max_depth=None, min_samples_split=2,
                min_samples_leaf=1, max_features='auto',
                bootstrap=True, oob_score=False, n_jobs=1,
                random_state=None, verbose=0)
              )]


    #print(X_train,y_train)
    #print(np.isnan(X_train).any())#判断是否有空值
    #print(np.isnan(y_train).any())
    # Iterate through the models
    for m in models:

        # Train each of the models on the training set
        m[1].fit(X_train, y_train)

        # Make an array of predictions on the test set
        pred = m[1].predict(X_test)

        # Output the hit-rate and the confusion matrix for each model
        print("%s:\n%0.3f" % (m[0], m[1].score(X_test, y_test)))
        #print(pred, y_test)
        print("%s\n" % confusion_matrix(y_test, pred, labels=[-1.0, 1.0]))#labels=["ant", "bird", "cat"]