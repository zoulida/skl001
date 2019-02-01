__author__ = 'zoulida'

import datetime

def strToDatetime(str):
    return
def datetimeToStr(datetimeForm, reform =  "%Y-%m-%d"):
    str = datetimeForm.strftime(reform)
    return str
def getDateStrNow():
    now = datetime.datetime.now()
    return datetimeToStr(now)
def getDateStrBefore(ndays = 100):
    now = datetime.datetime.now()
    before = now + datetime.timedelta(-ndays)
    return datetimeToStr(before)