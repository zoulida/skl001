__author__ = 'zoulida'

import  resource
import memory_profiler
def using():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    mem = usage[2]*resource.getpagesize() /1000000.0
    print("mem: ", mem,  " Mb")
    return mem


def mem_scan():
    before_mem = memory_profiler.memory_usage()

    for i in range(1000000):
        print(i)

    after_mem = memory_profiler.memory_usage()

    print("Memory (Before): {}Mb".format(before_mem))
    print("Memory (After): {}Mb".format(after_mem))

mem_scan()

using()

