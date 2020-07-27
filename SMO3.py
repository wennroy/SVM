# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 15:16:36 2020

@author: lengwaifang
"""

import numpy as np

def f(a,b):
    for i in range(len(x)):
        w += (a[i]*y[i]*x[i]).reshape(d,1)
    return np.matmul(x,w) + b
    
    
def update(a,b):
    ## 利用SMO计算w
    d = np.size(x,1)
    n = np.size(x,0)
    
    for i in range(d):
        for j in range(d):
            if i > j:
                continue
            eta = K[i][i] + K[j][j] - 2*K[i][j]
            Ei = f(a,b)[i] - y[i]
            Ej = f(a,b)[j] - y[j]
            
            


def SVM(N,C):
    d = np.size(x,1)
    n = np.size(x,0)
    
    w = np.random.rand(d,1)
    b = np.random.rand()
    a = np.random.rand(n,1)
    
    apoch = 0
    while apoch <=N:
        apoch += 1
        w, b = update(x,y,a,b)

def calculate_K(x):
    n = np.size(x,0)
    K = np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            K[i][j] = np.sum(x[i] * x[j])
    return K

if __name__ == '__main__':
    
    x = np.array([[4,3],
              [5,4],
              [4,5],
              [1,1],
              [2,4],
              [3,5],
              [4,2],
              [1,2],
              [2,1],
              [3,2]])
    y = np.array([1,1,1,-1,-1,1,-1,-1,-1,-1])
    
    K = calculate_K(x)