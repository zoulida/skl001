__author__ = 'Administrator'
import tushare as ts

#list = ts.get_hs300s()

#print(list)
#ts.set_token('69d6b836725cd75df21b39873603b14fed58d101bc033b991b51eb41')
#print(ts.__version__)


pro = ts.pro_api('69d6b836725cd75df21b39873603b14fed58d101bc033b991b51eb41')
#pro = ts.pro_api()
df = pro.daily(ts_code='000001.SZ', start_date='20180701', end_date='20180718')
print(df)