# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 20:18:46 2020

@author: xiaofangxiong
"""


import numpy as np
import matplotlib.pyplot as plt
from testCases import *
from planar_utils import load_test,plot_decision_boundary,load_planar_dataset, load_extra_datasets
np.random.seed(2)
class NN():
    def __init__(self,X,Y,hide_size = 4):
        self.x = X
        self.y = Y
        self.hide_size = 4
        self.w1 = np.random.randn(hide_size,X.shape[0])*0.01
        self.b1 = np.random.randn(hide_size,1)
        self.w2 = np.random.randn(1,hide_size)*0.01
        self.b2 = np.random.randn(1,1)
        self.m = X.shape[1]
    def print_shape(self):
        print('-'*14,'print shape','-'*13)
        print('the shape of w1{}'.format(self.w1.shape))
        print('the shape of b1{}'.format(self.b1.shape))
        print('the shape of w2{}'.format(self.w2.shape))
        print('the shape of b2{}'.format(self.b2.shape))
    def loss(self,Y_hat):
        
        log_loss = self.y*np.log(Y_hat) + (1-self.y)*np.log(1-Y_hat)
        return (-1/self.m)*np.sum(log_loss)
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    
    def forward_propagation(self):
        z1 = np.dot(self.w1,self.x) + self.b1
        a1 = np.tanh(z1)
        z2 = np.dot(self.w2,a1) + self.b2
        a2 = self.sigmoid(z2)
        return {
                'z1':z1,
                'a1':a1,
                'z2':z2,
                'a2':a2
                }

    def backward_propagation(self,z1,a1,z2,a2):
        dz2 = (1/self.m)*(a2 - self.y)
        dw2 = np.dot(dz2,a1.T)
        db2 = np.sum(dz2,axis=1,keepdims=True)
        dz1 = np.dot(self.w2.T,dz2)*(1-np.power(a1,2))
        dw1 = np.dot(dz1,self.x.T)
        db1 = np.sum(dz1,axis=1,keepdims=True)
        return {
                'dz2' : dz2,
                'dw2' : dw2,
                'db2' : db2,
                'dz1' : dz1,
                'dw1' : dw1,
                'db1' : db1
                }
    def update(self,par,lr = 0.01):
        self.w2 -= lr*par['dw2']
        self.w1 -= lr*par['dw1']
        self.b2 -= lr*par['db2']
        self.b1 -= lr*par['db1']
    def predict(self,x):
        z1 = np.dot(self.w1,x) + self.b1
        a1 = np.tanh(z1)
        z2 = np.dot(self.w2,a1) + self.b2
        a2 = self.sigmoid(z2)
        y_pre = np.zeros_like(a2)
        y_pre[a2>0.5] = 1
        y_pre[a2<0.5] = 0
        return y_pre
    def model(self,iteration = 2000,lr = 0.01,printloss = True):
        losses = []
        loss = 0
        for i in range(iteration):
            forward = self.forward_propagation()
            loss = self.loss(forward['a2'])
            par = self.backward_propagation(forward['z1'],forward['a1'],\
                                            forward['z2'],forward['a2'])
            self.update(par,lr=0.01)
            if i%100 == 0:
                losses.append(loss)
                if printloss:
                    print('the iterator:{},the loss:{}'.format(i,loss))
        y_pre = self.predict(self.x)
        accuracy = 1-(1/self.m)*np.sum(abs(y_pre-self.y))
        print('The number of iteration is {:d} and the accuracy of train is {:.6f}%'.format(iteration,accuracy*100))
        return losses,y_pre
    def test(self):
        x, y = load_test()
        y_pre = self.predict(x)
        accuracy = 1-(1/y.shape[1])*np.sum(abs(y_pre-y))
        print('the accuracy of test is {:.6f}%'.format(accuracy*100))
        return y,y_pre

        
if __name__ == '__main__':
    print('-'*15,'load data','-'*14)
    X,Y = load_planar_dataset()
    print('the shape of train data is {}\nthe shape of train label is {}'\
          .format(X.shape,Y.shape))
    print('-'*40)
    '''
    plt.scatter(X[0,:],X[1,:], c=Y.squeeze(), s=40, cmap=plt.cm.Spectral)
    plt.show()'''
    model = NN(X,Y,4)
    loss ,Y_pre = model.model(iteration=50000, lr=0.01)
    y,y_pre = model.test()
    plt.plot(loss)
    plt.show()
