__author__ = 'zoulida'

import numpy
import talib

close = numpy.random.random(100)
output = talib.SMA(close)
#print(output)

#from talib import MA_Type
#upper, middle, lower = talib.BBANDS(close, matype=MA_Type.T3)

output = talib.MOM(close, timeperiod=5)

import numpy as np
# note that all ndarrays must be the same length!
inputs = {
    'open': np.random.random(100),
    'high': np.random.random(100),
    'low': np.random.random(100),
    'close': np.random.random(100),
    'volume': np.random.random(100)
}
#print(inputs)
from talib import abstract
sma = abstract.SMA
sma = abstract.Function('sma')

input_arrays = inputs['close']
print(input_arrays)
#from talib.abstract import *
output = talib.SMA(input_arrays, timeperiod=25) # SMA均线价格计算收盘价
#output = talib.SMA(input_arrays, timeperiod=25, price='open') # SMA均线价格计算收盘价
upper, middle, lower = talib.BBANDS(input_arrays, 20, 2, 2)

#slowk, slowd = talib.STOCH(input_arrays, 5, 3, 0, 3, 0) # uses high, low, close by default
#slowk, slowd = talib.STOCH(input_arrays, 5, 3, 0, 3, 0, prices=['high', 'low', 'open'])

