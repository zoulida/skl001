__author__ = 'zoulida'
import numpy as np
from sklearn.model_selection import KFold
#from sklearn.cross_validation import KFold
#from sklearn import cross_validation
# dataset

#data = np.array([[1,3],[2,4],[3.1,3],[4,5],[5.0,0.3],[4.1,3.1]])
#label = np.array([0,1,1,1,0,0])
data = np.arange(24).reshape(12,2)
label = np.random.choice([1,2],12,p=[0.4,0.6])
sampNum= len(data)

# 10-fold （9份为training，1份为validation）
#kf = KFold(len(data), n_folds=4)
kf  = KFold(n_splits=5,shuffle=False)
#print(kf)
#iFold = 0
for train_index , test_index in kf.split(data):
#for train_index, val_index in kf:
    #iFold = iFold+1
    X_train, X_val, y_train, y_val = data[train_index], label[train_index], data[test_index],  label[test_index] # 这里的X_train，y_train为第iFold个fold的训练集，X_val，y_val为validation set
    print(X_train, X_val, y_train, y_val)
    print('======================================')

