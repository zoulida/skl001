__author__ = 'zoulida'

import Series

def test4():
    obj3 = Series(['blue', 'purple', 'yellow'], index=[0, 2, 4])
    print(obj3)
    obj4 = obj3.reset_index(drop=True)
    print(obj4)
    print( type(obj4))
