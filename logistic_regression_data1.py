# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 18:34:49 2020

@author: taylo
"""
#https://blog.csdn.net/m0_37867091/article/details/104887979
#昨天面试完普华的vi不想学习啊啊啊啊啊累死

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#import data
os.chdir(r'C:\Users\taylo\Desktop\ml\programming_exercise_for_ml\ml_ng\02-logistic_regression')
data=pd.read_csv('ex2data1.txt', names = ['Exam 1 score', 'Exam 2 score', 'Admitted or not admitted'])
# classify data
data.insert(0,'x_0',1) #insert x_0 column
col_num = data.shape[1] #extract number of column
print('There are ' + str(col_num) + ' columns in the dataset.')
x=data.iloc[:,0:col_num-1] #define x vector
y=data.iloc[:,col_num-1:col_num] #define y vector
print(x.shape) #extract dimension of vector x
print(y.shape) #extract dimension of vector y
#此时x和y的数据结构为DataFrame，需要转换为ndarray类型才能用于后续的矩阵操作
X=np.matrix(x.values)
Y=np.matrix(y.values)

#plot dataset
#sample test:
#x_1 = data.loc[data['Admitted or not admitted']==0].values #turn dataframe into ndarray
#print(x_1[:, 1:2])

a=data.loc[data['Admitted or not admitted']==0].values #turn dataframe into ndarray
b=data.loc[data['Admitted or not admitted']==1].values #turn dataframe into ndarray
fig,ax1 = plt.subplots()
ax1.scatter(data.loc[data['Admitted or not admitted']==0, 'Exam 1 score'], a[:,2], c='yellow', marker = 'o',label = 'Not_admitted')
#on the above line there are 2 ways expressing, cannot use index 1 to replace 'Exam 1 score'
#loc——通过行标签索引行数据, iloc——通过行号索引行数据 
ax1.scatter(b[:,1], b[:,2], c='black', marker = '+',label = 'Admitted')
ax1.legend()
ax1.set_xlabel('Exam 1 score')
ax1.set_ylabel('Exam 2 score')
plt.show()

#define logistic regression hypothesis

def sigmoid(X, theta):
    inner= X*theta
    return 1/(1+np.exp(-inner))

    
#define cost function
m=X.shape[0]
def costFunction(X, Y, theta):
    h=sigmoid(X, theta) #不能用h(x)命名当
    y_1=np.multiply(Y, np.log(h))
    y_2=np.multiply(1-Y, np.log(1-h))
    return np.sum(-y_1-y_2)/m
    
#define gradient descent
costs=[]
def gradientDescent(X, Y, theta, iters, alpha):
    for i in range(iters):
        h=sigmoid(X, theta)
        theta=theta-(alpha*X.T*(h-Y))/m
        cost=costFunction(X, Y, theta)
        costs.append(cost)
        if i%10000==0:
            print(cost)
    return theta, costs

theta=np.zeros((3,1))
iters=300000
alpha=0.009 
theta, costs = gradientDescent(X, Y, theta, iters, alpha)

#plot cost and iters，这个图一开始因为cost在0-6之间反复横跳，所以前面的挤一坨了
fig, ax2=plt.subplots()
ax2.plot([i for i in range(iters)],costs)
ax2.set_xlabel('iterations')
ax2.set_ylabel('cost')
plt.show()

#plot the prediction, since theta*X = 0 is the decision boundary, use the new hypothesis to calculate x2
X_1 = np.linspace(20,100,2)
X_2 = (-theta[0,0]/theta[2,0])+(-theta[1,0]/theta[2,0])*X_1
ax1.plot(X_1, X_2, label='Decision Boundary')
ax1.legend()
plt.show()

#check accuracy
def predict(X, theta):
    prob = sigmoid(X, theta)
    return [1 if result >= 0.5 else 0 for result in prob] #返回一个列表
h_x = predict(X,theta) 		#在参数为theta下的假设输出，为列表类型
h_x = np.array(h_x)					#将列表转化为一维数组
h_x = h_x.reshape(len(h_x),1)	#进而转化为二维数组，因为真实输出y也是二维数组，方便比较
acc = np.mean(h_x == y)#将假设输出和真实输出进行比较，求平均值
print(acc)
