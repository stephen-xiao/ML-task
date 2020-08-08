# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 10:40:23 2020

@author: xiaofangxiong
"""

import matplotlib.pyplot as plt
import numpy as np
import h5py
from dnn_app_utils_v2 import *
np.random.seed(1)
class NN():
    def __init__(self,X,Y,layer = 2,hide_size = None):
        self.x = X
        self.y = Y
        self.layer = layer
        if hide_size is None or len(hide_size) is not layer - 1:
            hide_size = [4]*(layer-1)
        self.hide_size = hide_size
        self.parameters = {}
        self.z = {}
        self.dz = {}
        self.da = {}
        self.a = {}
        self.m = X.shape[1]
    def initParameters(self):
        self.hide_size.append(1)
        self.parameters['w0'] = np.array(self.x)
        self.a[0] = np.array(self.x)
        for i in range(1,self.layer+1):
            self.parameters['w'+str(i)] = np.random.randn(self.hide_size[i-1],self.parameters['w'+str(i-1)].shape[0])/np.sqrt(self.parameters['w'+str(i-1)].shape[0])
            self.parameters['b'+str(i)] = np.zeros((self.hide_size[i-1],1))
    def printShape(self):
        print('-'*14,'print shape','-'*13)
        for i in range(1,self.layer+1):
            print('the shape of w{}:{}'.format(i,self.parameters['w'+str(i)].shape))
            print('the shape of b{}:{}'.format(i,self.parameters['b'+str(i)].shape))
    def loss(self,Y_hat):
        log_loss = self.y*np.log(Y_hat) + (1-self.y)*np.log(1-Y_hat)
        return (-1/self.m)*np.sum(log_loss)
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    def relu(self,z):
        return np.maximum(0,z)
    def linear(self,w,x,b):
        return np.dot(w,x)+b
    def forward_linear_activate(self,i,w,x,b,activate):
        self.z[i] = self.linear(w,x,b)
        if activate == 'sigmoid':
            return self.sigmoid(self.z[i])
        elif activate == 'relu':
            return self.relu(self.z[i])
        elif activate == 'tanh':
            return np.tanh(self.z[i])
    def forward_propagation(self):
        for i in range(1,self.layer+1):
            if i == self.layer:
                self.a[i] = self.forward_linear_activate(i,self.parameters['w'+str(i)],self.a[i-1],self.parameters['b'+str(i)],activate='sigmoid')
            else:
                self.a[i] = self.forward_linear_activate(i,self.parameters['w'+str(i)],self.a[i-1],self.parameters['b'+str(i)],activate='relu')
        return self.a[self.layer]
    def relugrad(self,cache):
        z = np.zeros_like(cache)
        z[cache>0] = 1
        z[cache<=0] = 0
        return z
    def sigmoidgrad(self,cache):
        z = self.sigmoid(cache)*(1-self.sigmoid(cache))
        return z
    def backward_propagation(self,lr = 0.0075):
        for i in range(self.layer,0,-1):
            if i == self.layer:
                self.da[i] = -(self.y/self.a[i] - (1-self.y)/( 1-self.a[i]))
                self.dz[i] = -(self.y/self.a[i] - (1-self.y)/( 1-self.a[i]))*self.sigmoidgrad(self.z[i])
                self.da[i] =  np.dot(self.parameters['w'+str(i)].T , self.dz[i])
            else: 
                self.dz[i] =  self.da[i+1]*self.relugrad(self.z[i])
                self.da[i] =  np.dot(self.parameters['w'+str(i)].T , self.dz[i])
            dw = np.dot(self.dz[i],self.a[i-1].T)
            db = np.dot(self.dz[i],np.ones((self.dz[i].shape[1],1)))
            self.parameters['w'+str(i)] -= (1/self.m)*lr*dw
            self.parameters['b'+str(i)] -= (1/self.m)*lr*db
    def predict(self,x):
        y_pre = np.zeros_like(x)
        y_pre[x>0.5] = 1
        y_pre[x<0.5] = 0
        return y_pre
    def model(self,iteration = 3000,lr = 0.0075,printloss = True):
        self.initParameters()
        self.printShape()
        losses = []
        loss = 0
        forward = 0
        for i in range(iteration):
            forward = self.forward_propagation()
            loss = self.loss(forward)
            self.backward_propagation()
            #print('the iterator:{},the loss:{:.6f}'.format(i,loss))
            if i%100 == 0:
                losses.append(loss)
                if printloss:
                    print('the iterator:{},the loss:{:.6f}'.format(i,loss))
        y_pre = self.predict(forward)
        accuracy = 1-(1/self.m)*np.sum(abs(y_pre-self.y))
        print('The number of iteration is {:d} and the accuracy of train is {:.6f}%'.format(iteration,accuracy*100))
        return losses,y_pre
    def test(self,x,y):
        x_pre = x
        x_next = x
        for i in range(1,self.layer+1):
            if i == self.layer:
                x_next = self.forward_linear_activate(i,self.parameters['w'+str(i)],x_pre,self.parameters['b'+str(i)],activate='sigmoid')
            else:
                x_next = self.forward_linear_activate(i,self.parameters['w'+str(i)],x_pre,self.parameters['b'+str(i)],activate='relu')
            x_pre = x_next
        y_pre = self.predict(x_next)
        accuracy = 1-(1/y.shape[1])*np.sum(abs(y_pre-y))
        print('the accuracy of test is {:.6f}%'.format(accuracy*100))
if __name__ == '__main__':
    print('-'*15,'load data','-'*14)
    X, Y, test_x_orig, test_y, classes = load_data()
    X = X.reshape(X.shape[0], -1).T   
    X = X/255
    test_x_orig = test_x_orig.reshape(test_x_orig.shape[0], -1).T  
    test_x_orig = test_x_orig/255
    print('the shape of train data is {}\nthe shape of train label is {}'\
          .format(X.shape,Y.shape))
    print('-'*40)
    nn = NN(X,Y,4,[20, 7, 5])
    #nn = NN(X,Y,2,[ 7 ])
    loss,y_pre = nn.model()
    nn.test(test_x_orig, test_y)
    plt.plot(loss)
    plt.show()


