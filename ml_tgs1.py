# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

iris = pd.read_csv("iris.csv")
iris = iris.head(100)

#inisialisasi
t1=0.14 #teta1
t2=0.39 #teta2
t3=0.55 #teta3
t4=0.72 #teta4
b=0.92 #bias
a=0.1 #learningrate
epoch=60 #epoch

#fungsi-fungsi
def h1(x1,x2,x3,x4,t1,t2,t3,t4,b):
    return (t1*x1)+(t2*x2)+(t3*x3)+(t4*x4)+b
    
def sigmoid_activation(x):
	return 1.0 / (1 + np.exp(-x))

def delta(x,pred,fact):
    return 2*(pred-fact)*(1-pred)*pred*x

def deltabias(pred,fact):
    return 2*(pred-fact)*(1-pred)*pred

def error1(pred,fact):
    return (pred-fact)*(pred-fact)

def teta(t,delta,a):
    #teta baru
    return t - (delta*a)

#cleaning data
#setosa = 1
#versicolor = 0
for x in range(0,100):
    if iris.iloc[x,4] in ['Iris-setosa']:
        iris.iloc[x,4]=1
    else:
        iris.iloc[x,4]=0

#inisialisasi
x1 = iris.iloc[:,0]
x2 = iris.iloc[:,1]
x3 = iris.iloc[:,2]
x4 = iris.iloc[:,3]
label = iris.iloc[:,4]
avg_error = pd.Series([])

#sgd epoch = 60
for current_epoch in range(0,epoch):
    h = h1(x1,x2,x3,x4,t1,t2,t3,t4,b)
    sigmoid = sigmoid_activation(h)
    error = error1(sigmoid,label)
    dteta1 = delta(x1,sigmoid,label)
    dteta2 = delta(x2,sigmoid,label)
    dteta3 = delta(x3,sigmoid,label)
    dteta4 = delta(x4,sigmoid,label)
    dbias = deltabias(sigmoid,label)
    t1 = teta(t1,dteta1,a)
    t2 = teta(t2,dteta2,a)
    t3 = teta(t3,dteta3,a)
    t4 = teta(t4,dteta4,a)
    b = teta(b,dbias,a)
    avg_error[current_epoch] = np.mean(error)

#grafik
fig = plt.figure()
plt.plot(np.arange(0, 60), avg_error, linestyle='-',marker='o')
fig.suptitle("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()