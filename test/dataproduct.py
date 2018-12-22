
import numpy as np
import matplotlib.pyplot as plt

# make_regression生成回归模型数据
from sklearn.datasets import make_regression

# 关键参数有n_samples（生成样本数），n_features（样本特征数），noise（样本随机噪音）和coef（是否返回回归系数
# X为样本特征，y为样本输出， coef为回归系数，共1000个样本，每个样本1个特征
X, y, coef = make_regression(n_samples=1000, n_features=1, noise=10, coef=True)

plt.scatter(X, y, c='b', s=3)
plt.plot(X, X * coef, c='r')
plt.xticks(())  # 不显示 x
plt.yticks(())  # 不显示 y
plt.show()

# make_classification生成三元分类模型数据
from sklearn.datasets import make_classification

# 关键参数有n_samples（生成样本数）， n_features（样本特征数）， n_redundant（冗余特征数）和n_classes（输出的类别数）
# X1为样本特征，Y1为样本类别输出， 共400个样本，每个样本2个特征，输出有3个类别，没有冗余特征，每个类别一个簇
X1, Y1 = make_classification(n_samples=400, n_features=2, n_redundant=0, n_clusters_per_class=1, n_classes=3)
plt.scatter(X1[:, 0], X1[:, 1], c=Y1, s=3, marker='o')
plt.show()

# make_blobs生成聚类模型数据
from sklearn.datasets import make_blobs

# 关键参数有n_samples（生成样本数）， n_features（样本特征数），centers(簇中心的个数或者自定义的簇中心)和cluster_std（簇数据方差，代表簇的聚合程度）
# X为样本特征，Y为样本簇类别， 共1000个样本，每个样本2个特征，共3个簇，簇中心在[-1,-1], [1,1], [2,2]， 簇方差分别为[0.4, 0.5, 0.2]
X, y = make_blobs(n_samples=1000, n_features=2, centers=[[-1, -1], [1, 1], [2, 2]], cluster_std=[0.4, 0.5, 0.2])
plt.scatter(X[:, 0], X[:, 1], c=y, s=3, marker='o')
plt.show()

# make_gaussian_quantiles生成分组多维正态分布的数据
from sklearn.datasets import make_gaussian_quantiles

# 关键参数有n_samples（生成样本数）， n_features（正态分布的维数），mean（特征均值）， cov（样本协方差的系数）， n_classes（数据在正态分布中按分位数分配的组数）
# 生成2维正态分布，生成的数据按分位数分成3组，1000个样本,2个样本特征均值为1和2，协方差系数为2
X1, Y1 = make_gaussian_quantiles(n_samples=1000, n_features=2, n_classes=3, mean=[1, 2], cov=2)
plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1, s=3)
plt.show()
