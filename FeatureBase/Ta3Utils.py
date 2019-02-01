__author__ = 'zoulida'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 16:36:02 2018

@author: lg
"""

import tushare as ts
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import talib

from talib import *

df=ts.get_k_data('600600')


# time period of rsi normaly in [10,15,30,60]
talib.RSI(df.open, timeperiod=12)

df['k'],df['d']=talib.STOCH(df.high,df.low,df.close, fastk_period=9,slowk_period=3,slowk_matype=0,slowd_period=3,slowd_matype=0)

df['j']=3*df.k-2*df.d

#量价指标
real = talib.OBV(df.close, df.volume)
real = talib.AD(df.high, df.low, df.close, df.volume)
real = talib.ADOSC(df.high, df.low, df.close, df.volume, fastperiod=3, slowperiod=10)
#avg




real = talib.DEMA(df.close, timeperiod=30)

real = talib.EMA(df.close, timeperiod=30)

real = talib.HT_TRENDLINE(df.close)

real = talib.KAMA(df.close, timeperiod=30)

##################################################
real = talib.SAREXT(df.high, df.low, startvalue=0, offsetonreverse=0, accelerationinitlong=0,
                      accelerationlong=0, accelerationmaxlong=0, accelerationinitshort=0, accelerationshort=0, accelerationmaxshort=0)




real = talib.T3(df.close, timeperiod=5, vfactor=0)
#moment
real = talib.ADX(df.high, df.low, df.close, timeperiod=14)

real = talib.APO(df.close, fastperiod=12, slowperiod=26, matype=0)

real = talib.AROONOSC(df.high, df.low, timeperiod=14)

real = talib.BOP(df.open, df.high, df.low, df.close)

real = talib.CCI(df.high, df.low, df.close, timeperiod=14)

real = talib.CMO(df.close, timeperiod=14)

real = talib.MFI(df.high, df.low, df.close, df.volume, timeperiod=14)

real = talib.MINUS_DI(df.high, df.low, df.close, timeperiod=14)

real = talib.ROC(df.close, timeperiod=10)
fastk, fastd = talib.STOCHF(df.high, df.low, df.close, fastk_period=5, fastd_period=3, fastd_matype=0)

real = talib.TRIX(df.close, timeperiod=30)

#****************************
real = talib.ULTOSC(df.high, df.low, df.close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
real = talib.WILLR(df.high, df.low, df.close, timeperiod=14)

#波动率指标函数
real = talib.ATR(df.high, df.low, df.close, timeperiod=14)

real = talib.NATR(df.high, df.low, df.close, timeperiod=14)

real = talib.TRANGE(df.high, df.low, df.close)

#周期
real = talib.HT_DCPERIOD(df.close)

real = talib.HT_DCPHASE(df.close)

inphase, quadrature = talib.HT_PHASOR(df.close)

sine, leadsine = talib.HT_SINE(df.close)

#integer = talib.HT_TRENDMODE(df.close)

#integer = talib.CDL2CROWS(df.open, df.high, df.low, df.close)


#统计学指标

real = talib.BETA(df.high, df.low, timeperiod=5)
real = talib.CORREL(df.high, df.low, timeperiod=30)
#************************

#timeperiod=3,5,14,30,
real = talib.LINEARREG_SLOPE(df.close, timeperiod=3)

real = talib.STDDEV(df.close, timeperiod=5, nbdev=1)
real = talib.TSF(df.close, timeperiod=14)

#real = talib.ACOS(df.close)
