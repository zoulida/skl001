__author__ = 'zoulida'
import pandas as pd
import numpy as np

from pandas import Series,DataFrame

df1 = pd.DataFrame(np.random.randn(4,4),index=list('ABCD'),columns=list('ABCD'))
df2 = pd.DataFrame(np.random.randn(4,4),index=list('ABCD'),columns=list('ABCD'))

dict = {'Alice': df1, 'Beth': df2, 'Cecil': '3258'}
print(dict['Alice'])
print(dict['Beth'])
print(dict.get('Cecil'))
print(dict.get('Beth1'))

dict['age'] = 16
print(dict.get('age'))
dict['age'] = 17
print(dict.get('age'))
dict1 = {}
print(dict1.get('age'))