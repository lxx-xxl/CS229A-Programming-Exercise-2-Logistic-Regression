# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 16:40:20 2020

@author: taylo
"""
#https://blog.csdn.net/m0_37867091/article/details/104887979
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import data
os.chdir(r'C:\Users\taylo\Desktop\ml\programming_exercise_for_ml\ml_ng\02-logistic_regression')
data=pd.read_csv('ex2data2.txt', names = ['Microchip Test 1', 'Microchip Test 2', 'Result'])

#plot dataset
fig,ax1 = plt.subplots()
y0=data.loc[data['Result']==0].values
y1=data.loc[data['Result']==1].values
ax1.scatter(y0[:,0], y0[:,1], marker='o', c='yellow', label = 'y =0 ')
ax1.scatter(y1[:,0], y1[:,1], marker='+', c='black', label = 'y =1 ')
#经过特征映射后的X已经具有x1^0*x2^0 = 1 = x0, x_1^0x_2^0=1=x_0这一项，因此不需要再单独插入x0 = 1 
ax1.legend()
ax1.set(xlabel = 'Microchip Test 1', ylabel = 'Microchip Test 2')
plt.show()

#从上图可以看出，该逻辑回归的样本数量多，且分布是非线性的，而原始特征只有x1,x2给分类带来不小难度。
#解决办法是用多项式创建更多的特征,因为更多的特征进行逻辑回归时，得到的决策边界可以是任意高阶函数的形状。
#https://blog.csdn.net/m0_37867091/article/details/104887979，这个function是复制的
def feature_mapping(x1,x2,power):
    data_new = {} # 字典 保存特征多项式
    for i in np.arange(power+1):  # i的范围是(0,power]，整数
        for j in np.arange(i + 1):# j的范围是(0,i]，整数
            data_new['F{}{}'.format(i-j,j)] = np.power(x1,i-j) * np.power(x2,j)    
    return pd.DataFrame(data_new) 

# 获取原始特征作为 feature_mapping 的参数
x1 = data['Microchip Test 1']
x2 = data['Microchip Test 2']
# 得到特征映射后的特征集
power = 6
data_map = feature_mapping(x1,x2,power)

#classify data
col_num = data.shape[1]
print('There are ' + str(col_num) + ' columns in the dataset.')
x=data.iloc[:,0:col_num-1] #define x vector
y = data.iloc[:,col_num-1:col_num]
X=np.matrix(data_map.values)
Y=np.matrix(y.values)
print(X.shape)
print(Y.shape)
theta = np.zeros((X.shape[1],Y.shape[1]))
print(theta.shape)
#define regularized logistic regression hypothesis
#由于特征映射后维数较高，模型更复杂，容易产生过拟合现象，因此需要对代价函数正则化(Regularize)
def sigmoid(X, theta):
    inner= X*theta
    return 1/(1+np.exp(-inner))

#define regularized cost function
def costfunction(X, Y, theta, lamda):
    h=sigmoid(X, theta)
    y_1=np.multiply(Y, np.log(h))
    y_2=np.multiply(1-Y, np.log(1-h))
    reg = np.multiply(np.sum(np.power(theta[1:], 2)), lamda/2*len(X))
    #注意正则化项的θ是从j=1开始
    return np.sum(-y_1-y_2)/len(X)+reg

#define regularized gradient descent
costs=[]
def gradientDescent(X, Y, theta, lamda, alpha, iters):
    for i in range(iters):
        h=sigmoid(X, theta)
        first = (X.T*(h-Y))/len(X)
        second = lamda*theta[1:]/len(X)
        # 再在首项插入0，使reg维度为(28,1)与θ一致。axis=0表示按行插入
        second = np.insert(second,0,values=0,axis=0) 
        theta = theta-alpha*(first+second)
        cost=costfunction(X, Y, theta, lamda)
        costs.append(cost)
        if i % 10000 == 0:
            print(cost)
    return theta, costs

#assign value
alpha = 0.001
iters = 200000
lamda = 0.001
theta_final,costs = gradientDescent(X, Y, theta, lamda, alpha, iters) 
        
#plot cost and iters
fig,ax2 = plt.subplots()
ax2.scatter([i for i in range(iters)], costs)
ax2.set_xlabel('iterations')
ax2.set_ylabel('cost')
plt.show()

#visualize regularized decision boundary，这一坨都是复制的
fig,ax1 = plt.subplots()
y0=data.loc[data['Result']==0].values
y1=data.loc[data['Result']==1].values
ax1.scatter(y0[:,0], y0[:,1], marker='o', c='yellow', label = 'y =0 ')
ax1.scatter(y1[:,0], y1[:,1], marker='+', c='black', label = 'y =1 ')
a = np.linspace(-1.2,1.2,200)
xx,yy = np.meshgrid(a,a) #生成200×200的网格采样点
z = feature_mapping(xx.ravel(),yy.ravel(),6).values # 特征映射得到多维特征 z ,ravel()将多维数组转换为一维数组,6与前面power对应
zz = z*theta_final
zz = zz.reshape(xx.shape)
plt.contour(xx,yy,zz,0)# ?Xθ = 0 （也就是104行）即为决策边界，contour()用于绘制等高线
ax1.legend()
plt.show()

#check accuracy
def predict(X, theta):
    prob = sigmoid(X, theta)
    return [1 if result >= 0.5 else 0 for result in prob] #返回一个列表
h_x = predict(X,theta_final) 		#在参数为theta下的假设输出，为列表类型
h_x = np.array(h_x)					#将列表转化为一维数组
h_x = h_x.reshape(len(h_x),1)	#?进而转化为二维数组，因为真实输出y也是二维数组，方便比较
acc = np.mean(h_x == y)#将假设输出和真实输出进行比较，求平均值
print(acc)
