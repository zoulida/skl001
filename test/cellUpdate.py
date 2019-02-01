__author__ = 'zoulida'
import pandas as pd
import numpy as np

df = pd.DataFrame(np.arange(12).reshape(3,4), columns=[chr(i) for i in range(97,101)])
print(df)

df.iloc[1,3] = '老王'
print(df)
df.loc[2,'d'] = '你好'
print(df)

df = df.sort_values(by = 'a',axis = 0,ascending = False)
print(df)