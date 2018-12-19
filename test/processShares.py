__author__ = 'zoulida'

#coding: utf-8
from multiprocessing import Process, Manager
# 每个子进程执行的函数
# 参数中，传递了一个用于多进程之间数据共享的特殊字典
def func(i, d):
    d[i] = i + 100
    print(d.values())


if __name__ == '__main__':
    # 在主进程中创建特殊字典
    m = Manager()
    d = m.dict()
    for i in range(50):
        # 让子进程去修改主进程的特殊字典
        p = Process(target=func, args=(i, d))
        p.start()
    p.join()
