# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 15:20:05 2020

@author: lengwaifang
"""


def parameter_eclipse(w):
    '''
    Input: w1x1^2 + w2x2^2 + w3x1 +w4x2 + w5x1x2 +w0 =0
    Output: mu1, mu2, width, height, angle
    '''
    import numpy as np
    import math
    sigma1 = math.sqrt(1/abs(w[1]))
    sigma2 = math.sqrt(1/abs(w[2]))
    rho12 = -w[5]/2*sigma1*sigma2
    if not rho12 == 0:
        mu1 = (w[4] + w[3]/sigma2/rho12)/2/(rho12/sigma1-1/(sigma1**2*rho12*sigma2))
        mu2 = mu1*sigma2/(sigma1**2*rho12) + w[3]*sigma2/(2*rho12)
    else:
        mu1 = -sigma1**2*w[3]/2
        mu2 = -sigma2**2*w[4]/2
    c2 = mu1**2/sigma1**2 + mu2**2/sigma2**2 - 2*mu1*mu2*rho12/(sigma1*sigma2)-w[0]
    if c2<=0:
        print('不是椭圆')
        return None
    else:
        c = math.sqrt(c2)
    # return sigma1,sigma2,mu1,mu2,rho12,c
        
    S = np.matrix([[sigma1**2,rho12*sigma1*sigma2],[rho12*sigma1*sigma2,sigma2**2]])
    # np.linalg.eig(S)
    # return S
    eig = np.linalg.eig(S)
    angle = math.atan(eig[1][0].reshape(-1,1)[0]/eig[1][0].reshape(-1,1)[1])/(2*math.pi)*360
    height = math.sqrt(eig[0][0])
    width = math.sqrt(eig[0][1])
    
    return mu1, mu2, width, height, angle

    # from matplotlib.patches import Ellipse
    # from matplotlib import pyplot as plt
    # e = Ellipse(xy = (mu1,mu2), height = height*2, width =width * 2, angle=angle)
    # fig = plt.figure(0)
    # ax = fig.add_subplot(111, aspect='equal')
    # ax.add_artist(e)
    # e.set_facecolor("white")
    # plt.xlim(-10, 10)
    # plt.ylim(-10, 10)
    # ax.grid(True)
    # plt.title("50% Probablity Contour - Homework 4.2")
