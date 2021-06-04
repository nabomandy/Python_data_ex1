# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 16:18:34 2020

@author: sist
"""
import numpy as np

LEARNING_RATE=0.01
weight=0
bias=1.0
x = np.array([[2.,-1.],[1.,-2.]])
y = np.array([2.,-6.])


def backprop_postproc(G_loss, diff):
    return 2*diff / np.prod(diff.shape)


def backprop_neuralnet(G_output, x):
    global weight, bias
    g_output_w = x.transpose() # 행과열을 바꾼다 
    
    # #행령의 곱 
    #print("x:{}, weight:{},bias:{}".format(x.shape, weight.shape, bias.shape))
    #print(bias)
    G_w = np.dot(g_output_w, G_output)
    G_b = np.sum(G_output, axis=0)
    weight -= LEARNING_RATE * G_w
    bias -= LEARNING_RATE * G_b
    #print("w:", weight)
    print(weight,":",bias)

def forward_neuralnet(x):
    global weight, bias
    #식의 값을 계산한다 
    output = np.dot(x, weight) + bias
    return output, x

def forward_postproc(output, y):
    print(output.shape,":",y.shape)
    diff = output - y
    #np.square :제곱값
    square = np.square(diff)
    #np.mean : 전체성분의 평균값
    loss = np.mean(square)
    #loas는 오차곱 평균 이고 diff는 계산값
    return loss, diff

b = 1
y1=np.dot(x,weight)+b
print(y1)


for i in range(10) :
    print(i,"==============")
    output, x = forward_neuralnet(x)
    loss, diff = forward_postproc(output, y)
    G_output=backprop_postproc(0.1, diff)
    backprop_neuralnet(G_output, x)
    print("w:",weight)
    print("b:",bias)

y3=np.dot(x[0],weight)+bias
print(y3[0],":",y[0])


