__author__ = 'zoulida'

from multiprocessing import Pool,Process,Value,Array,Manager
def f(n,a):
    n.value = 888
    for i in range(10):
        a[i] = -a[i]


def f2():
    num.value = 888
    for i in range(10):
        arr[i] = -arr[i]

if __name__ == '__main__':
    num = Value('d',0.0)
    arr = Array('i',range(10))

    print(num.value)
    print(arr[:])

    #p = Process(target=f2,args=())
    p = Process(target=f,args=(num,arr))
    p.start()
    p.join()  #窜行
    print(num.value)
    print(arr[:])
