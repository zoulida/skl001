__author__ = 'zoulida'

import StorckList.ConditionList as CT
def getMiddleList():
    #import StorckList
    CL = CT.ConditionLists()
    CL.getDFHS300andADV()
    df = CL.dataAndADV
    frame = df.sort_values(by = 'ADV',axis = 0,ascending = True)
    #print(frame)

    returnData = frame[round(frame.shape[0]/3):round(frame.shape[0]*2/3)]
    #print(returnData)
    return returnData

def computeScore():
    list = getMiddleList()
    list['Score'] = 0
    list.set_index('Code', inplace=True)
    #print(list)
    for index, row in list.iterrows():
        print('开始预测证券： ', index)
        import forcastPa.HS300Class_CrossValidationDEF as CCD
        score = CCD.classMain(index)
        #print(score)
        list.loc[index, 'Score'] = score
        #list.iloc[]
    list = list.sort_values(by = 'Score',axis = 0,ascending = False)
    print(list)
    return list
if __name__ == "__main__":
    computeScore()