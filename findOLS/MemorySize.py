__author__ = 'zoulida'


import memory_profiler



def mem_scan():
    before_mem = memory_profiler.memory_usage()

    #for i in range(1000000):
    #    print(i)

    after_mem = memory_profiler.memory_usage()

    print("Memory (Before): {}Mb".format(before_mem))
    print("Memory (After): {}Mb".format(after_mem))

    import sys as sys

    a = [x for x in range(10000)]
    print ('对象大小 ', sys.getsizeof(a))

mem_scan()

#using()

