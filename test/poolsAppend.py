__author__ = 'zoulida'
from multiprocessing import Pool
import time,random,os

def work(n):
    time.sleep(1)
    return n**2
if __name__ == '__main__':
    p=Pool()

    res_l=[]
    for i in range(10):
        res=p.apply_async(work,args=(i,))
        res_l.append(res)

    p.close()
    p.join() #等待进程池中所有进程执行完毕

    nums=[]
    for res in res_l:
        nums.append(res.get()) #拿到所有结果
    print(nums) #主进程拿到所有的处理结果,可以在主进程中进行统一进行处理