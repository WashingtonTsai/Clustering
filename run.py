# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 11:29:37 2016
Run Kmeans classifier
@author: liudiwei
"""
import pandas as pd
import numpy as np
from kmeans import KMeansClassifier
import matplotlib.pyplot as plt
import xlrd
from sklearn import metrics
from mpl_toolkits.mplot3d import Axes3D

cValue = ['cyan','darkgreen','darkgray','darksalmon','darkred','olive','yellow','yellowgreen',
'silver','red','purple','pink','orangered','orange','navy','magenta','lightgoldenrodyellow',
'lavenderblush','honeydew','mediumseagreen']  

def readUCIIris():
    data = []
    # read dataset
    raw = pd.read_csv('iris.csv')
    raw_data = raw.values
    raw_feature = raw_data[0:,0:4]
    for i in range(len(raw_feature)):
        ele = []
        ele.append(list(raw_feature[i]))
        if raw_data[i][4] == 'Iris-setosa':
           #ele.append([1,0,0])
            ele.append(0.0)
        elif raw_data[i][4] == 'Iris-versicolor':
            #ele.append([0,1,0])
            ele.append(1.0)
        else:
            #ele.append([0,0,1])
            ele.append(2.0)
        data.append(ele)

    # print data
    # 随机排列data
    np.random.shuffle(data)
    # print data
    training = data[0:100]
    test = data[101:]
    return training,test

def splitlabanddata(trainingdata):
    trainingfeature = []
    traininglabel = []
    for singlesample in trainingdata:
        trainingfeature.append(singlesample[0])
        traininglabel.append(singlesample[1])
    return trainingfeature,traininglabel

def drawcluter_result(data,colorlist,sse,ARI):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:,1], data[:,2], zs=data[:,3], c=colorlist, depthshade=False)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    #plt.title("SSE={:.2f}".format(sse))
    plt.title("SSE={:.2f},ARI={:.4f}".format(sse,ARI))
    plt.show()


def drawcluter_real(data,colorlist):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:,1], data[:,2], zs=data[:,3], c=colorlist, depthshade=False)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

def readUCIdata1(filename):
    xls_file = xlrd.open_workbook(filename)
    xls_sheet = xls_file.sheets()[0] 
    row_allnum = 29  #28 data + 1 class
    col_allnum = 20  #20 class fragrance
    labellist = []
    datalist = [] 
    labelreallist = []
    datalistx = []
    datalisty = []
    drawcolorlist = []
    col_value = xls_sheet.col_values(0)
    for i in range(col_allnum):
        labellist.append(str(col_value[i]))
    for j in range(col_allnum):
        row_value = xls_sheet.row_values(j) 
        for i in range(1,row_allnum):
            firstattribute = float(int(row_value[i]/1000))
            secondattribute = float(int(row_value[i] - firstattribute*1000))
            datalist.append([firstattribute,secondattribute])
            datalistx.append(firstattribute)
            datalisty.append(secondattribute)
            labelreallist.append(j)
            drawcolorlist.append(cValue[j]) 
    return datalist,labelreallist,labellist,drawcolorlist,datalistx,datalisty



def displayUCIdata(filename):
    datalist,labelreallist,labellist,drawcolorlist,datalistx,datalisty = readUCIdata(filename)
    fig = plt.figure()  
    ax1 = fig.add_subplot(111)  
    ax1.set_title('Perfume Scatter Plot')  
    plt.xlabel('X')  
    plt.ylabel('Y')  
    ax1.scatter(datalistx,datalisty,c=drawcolorlist,marker='s')  
    #plt.legend('x1')  
    plt.show()


#加载数据集，DataFrame格式，最后将返回为一个matrix格式
def loadDataset(infile):
    df = pd.read_csv(infile, sep='\t', header=0, dtype=str, na_filter=False)
    return np.array(df).astype(np.float)

def readUCIdata():
    datalist = []
    labellist = []
    datatxt = open("wine.txt")
#lines  = datatxt.readlines()
    line = datatxt.readline()
    i = 1 
    while line:
        linestemp = line.strip('.\n').split(',')
        #print(linestemp[len(linestemp)-1] )
        #labellist.append(linestemp[len(linestemp)-1])
        for i in range(1,len(linestemp)):
            linestemp[i] = float(linestemp[i])
        linestemp1 = linestemp[1:(len(linestemp)-1)]
        datalist.append(linestemp1)  
        labellist.append(int(linestemp[0]))
        line = datatxt.readline()
    return np.array(datalist),np.array(labellist)


def ARIcaculate(labels_true,labels_pred):
    #labels_true = [1,1,0,0,0,1,1,1]
    #labels_pred = [0,0,1,1,1,0,0,0]
    ARI = metrics.adjusted_rand_score(labels_true,labels_pred)
    #print(ARI)
    return ARI


if __name__=="__main__":
    #data_X = loadDataset(r"data/testSet.txt")
    #data_X,label_X = readUCIdata()
    #data_X,labelreallist,labellist,drawcolorlist,datalistx,datalisty = readUCIdata1('perfume_data.xlsx')
    #print(data_X[0][1])
    trainingdata,testdata = readUCIIris()
    trainingfeature,traininglabel = splitlabanddata(trainingdata)
    
    data_X = np.array(trainingfeature)
    #k = 4
    k=3
    print(data_X)
    #print(data_X[0])
    clf = KMeansClassifier(k)
    clf.fit(data_X)
    cents = clf._centroids
    labels = clf._labels
    sse = clf._sse
    colors = ['red','purple','darkgreen','darkgray','darksalmon','darkred','olive','yellow','yellowgreen',
'silver','cyan','pink','orangered','orange','navy','magenta','lightgoldenrodyellow',
'lavenderblush','honeydew','mediumseagreen']  

    
    print(cents)
    pred =  clf.predict(data_X)
    print(pred)
    print("The labels is:",labels)
    print(traininglabel)
    colorlist = []
    colorlistreal = []
    ARI = ARIcaculate(traininglabel,pred)
    for i in range(len(data_X)):
       colorlist.append(colors[int(pred[i])])
       colorlistreal.append(colors[int(traininglabel[i])])
    drawcluter_result(data_X,colorlist,sse,ARI)
    drawcluter_real(data_X,colorlistreal)
    #print(sse)
    plt.title("SSE={:.2f},ARI={:.4f}".format(sse,ARI))
    plt.axis([-7,7,-7,7])
    outname = "./result/k_clusters" + str(k) + ".png"
    plt.savefig(outname)
    plt.show()
    
'''
    #colors = ['b','g','r','k','c','m','y','#e24fff','#524C90','#845868']
    for i in range(k):
        index = np.nonzero(labels==i)[0]
        print("Index is:",index)
        x0 = data_X[index, 1]
        x1 = data_X[index, 3]
        y_i = i
        for j in range(len(x0)):
            #pass
            plt.text(x0[j], x1[j], str(y_i), color=colors[i], \
                        fontdict={'weight': 'bold', 'size': 6})
        plt.scatter(cents[i,0],cents[i,1],marker='x',color=colors[i],\
                    linewidths=7)
'''
    
