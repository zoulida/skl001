__author__ = 'zoulida'
import heapq
import datetime
import pandas as pd

def get_least_numbers_big_data( alist, k):
    max_heap = []
    length = len(alist)
    if not alist or k <= 0 or k > length:
        return
    k = k - 1

    for ele in alist:
        start = datetime.datetime.now()
        ele = -ele
        if len(max_heap) <= k:
            heapq.heappush(max_heap, ele)#往堆中插入一条新的值
        else:
            heapq.heappushpop(max_heap, ele)#顾名思义，将值插入到堆中同时弹出堆中的最小值
        end = datetime.datetime.now()
        print('消耗时间 ' , (end - start).total_seconds())
    return map(lambda x:-x, max_heap)

def get_least_row():
    b = [('a',1),('b',2),('c',3),('d',4),('e',5)]
    list = heapq.nlargest(1,b,key=lambda x:x[1])
    heap = []
    #向推中插入元组
    heapq.heappush(heap,(10,'ten'))


    insertRow1 = pd.DataFrame([['4444', "55555", 51]], columns=['A', 'B', 'adf'])
    insertRow2 = pd.DataFrame([['4444', "55555", 52]], columns=['A', 'B', 'adf'])
    insertRow3 = pd.DataFrame([['4444', "55555", 53]], columns=['A', 'B', 'adf'])

    insertRow1 = insertRow1.append([('4444', "55555", 52)])
    list2 = [insertRow1,insertRow2,insertRow3]

    print(insertRow1)

if __name__ == "__main__":
    get_least_row()
    import random
    L=[random.randint(0,10000) for _ in range(500) ]
    l = [1, 9, 2, 4, 7, 6, 3]
    min_k = get_least_numbers_big_data(l, 3)
    print([x for x in min_k])