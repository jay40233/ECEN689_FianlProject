"""
@author : Abhijeet Kumar
Created on July 02, 2015
comments : Module for reading a csv file and 
           returning training and test data           
           Module for writing csv file
"""

import csv
import numpy as np

def getTraindata(filename):
    with open(filename,'rb') as f:     
        reader = csv.reader(f)
        Y = None
        X = []
        for temp in reader:
            try:
              row = [float(n) for n in temp]
              if Y == None:
                Y = row[:1]
              else:
                Y = np.hstack((Y,row[:1]))
              X.append(row[1:])
            except:
              continue
        L = np.asarray(Y)
        X = np.asarray(X)
        return X,L       

    
def getTestdata(filename):
    with open(filename,'rb') as f:     
        reader = csv.reader(f)
        T = []
        for temp in reader:
            try:
              row = [float(n) for n in temp]
              T.append(row)
            except:
              continue
        T = np.asarray(T)
        return T
def getSolution(filename):
    with open(filename,'rb') as f:     
        reader = csv.reader(f)
        T = []
        for temp in reader:
            try:
              row = int(temp[0])
              T.append(row)
            except:
              continue
        T = np.asarray(T)
        return T

def writecsv(result):
    with open('result.csv','wb') as fp:
        a = csv.writer(fp,delimiter =',')
        header = ['Result']
        result = np.hstack((header,result))
        a.writerows(result)
        

            
if __name__ == "__main__":
    print "get Train and Test data from this module"
    
    