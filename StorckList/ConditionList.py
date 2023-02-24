__author__ = 'zoulida'
import pandas as pd

class ConditionLists:#类范例
    #list = None
    def __init__(self):
        return

    def getList(self, setName = 'HS300'):
        import tushare as ts
        if(setName == 'HS300'):
            self.list = ts.get_hs300s()
        print(self.list)
        return self.list

    def getDFHS300(self):
        data = pd.DataFrame()# 初始化dataframe为空值
        list = self.getList(setName = 'HS300')
        for index, row in list.iterrows():
            print (row["code"])
            import DBStock.dbQueryPools as dbP
            df = dbP.queryMySQL_getLast(row["code"])
            #if(data.__len__() == 0):
            #    data = df
            data = data.append(df)
        #print(data.head(12))
        self.data = data
        return self.data

    def getDFHS300andADV(self):
        data = pd.DataFrame()# 初始化dataframe为空值
        list = self.getList(setName = 'HS300')
        for index, row in list.iterrows():
            df = self.getLastOneAndADV(row["code"])
            data = data.append(df)
        self.dataAndADV = data
        print(data)
        return self.dataAndADV

    def getLastOneAndADV(self, code):
        import DBStock.dbQueryPools as dbpool
        import toolsproject.timeTool as tT
        #print(tT.getDateStrNow(), tT.getDateStrBefore())
        ts = dbpool.queryMySQL_plot_stock_market(code, tT.getDateStrBefore(), tT.getDateStrNow())
        #print(ts)
        ts['Date'] = pd.to_datetime(ts['Date'])
        ts.set_index('Date', inplace=True)
        ts = ts[(True^ts['Close'].isin([0]))]#条件删除去除值为0的行
        import FeatureBase.FeatureUtils as FU
        ts = FU.ADV(ts)
        #打印不换行
        #pd.set_option('display.height',1000)
        #pd.set_option('display.max_rows',500)
        #pd.set_option('display.max_columns',500)
        pd.set_option('display.width',1000)
        #print(ts.tail(1))
        #self.dataAndADV = ts
        return ts.tail(1)

if __name__ == "__main__":
    CL = ConditionLists()
    #CL.getDFHS300()
    #print(CL.data)

    #CL.getLastOneAndADV('600016')
    CL.getDFHS300andADV()
    df = CL.dataAndADV
    #frame = df.sort(columns = ['ADV'],axis = 0,ascending = True)
    frame = df.sort_values(by = 'ADV',axis = 0,ascending = True)
    print(frame)