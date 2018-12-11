import pandas as pd
import numpy as np


df = pd.DataFrame(np.random.randn(8,5))
print(df)
#insertRow = pd.DataFrame([[0.,0.,0.,0.,0.]],columns = ['date','spring','summer','autumne','winter'])
insertRow = pd.DataFrame([[1.,2.,3.,4.,5.]])
print(insertRow)
above = df.loc[:2]
print(above)
below = df.loc[3:]
print(below)
newData = above.append(insertRow,ignore_index=False)
print(newData)
