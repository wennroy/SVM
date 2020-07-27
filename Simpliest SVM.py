# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 16:12:39 2020

@author: lengwaifang
"""


import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm

    
    


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
    
    clf = svm.LinearSVC(C=1e10)
    clf.fit(x, y) 
    w = clf.coef_
    b = clf.intercept_
    w = np.hstack((b.reshape(1,1),w)).reshape(np.size(x,1)+1)
    
    xmin, xmax = np.min(x[:,0]), np.max(x[:,0])
    xw = np.linspace(xmin,xmax,50)
    yw = (-w[0]-w[1]*xw)/w[2]
    plt.plot(xw,yw,'b-')
    plot_svc_decision_boundary(clf,axis=([xmin-1,xmax+1,1-1,6+1]))
    plt.scatter(x[:,0],x[:,1],c = y)
    plt.show()
    

'''
https://blog.csdn.net/qq_16953611/article/details/82414129             sklearn-SVC参数
https://www.cnblogs.com/Yanjy-OnlyOne/p/11368253.html                   某人的代码，看起来很牛逼
'''
def plot_svc_decision_boundary(model,axis):
    x0,x1=np.meshgrid(
        np.linspace(axis[0],axis[1],int((axis[1]-axis[0])*100)).reshape(-1,1),
        np.linspace(axis[2],axis[3], int((axis[3] - axis[2]) * 100)).reshape(-1,1)
    )
    x_new=np.c_[x0.ravel(),x1.ravel()]
    y_pre=model.predict(x_new)
    zz=y_pre.reshape(x0.shape)
    from matplotlib.colors import ListedColormap
    cus=ListedColormap(["#EF9A9A","#FFF59D","#90CAF9"])
    plt.contourf(x0,x1,zz,cmap=cus)
    w=model.coef_[0]
    b=model.intercept_[0]
    x1=np.linspace(axis[0],axis[1],200)
    upy=-w[0]*x1/w[1]-b/w[1]+1/w[1]
    downy=-w[0]*x1/w[1]-b/w[1]-1/w[1]
    upindex=((upy>axis[2])&(upy<axis[3]))
    downindex = ((downy > axis[2]) & (downy < axis[3]))
    plt.plot(x1[upindex],upy[upindex],"r")
    plt.plot(x1[downindex],downy[downindex],"g")
# plot_svc_decision_boundary(s11,axis=([-3,3,-3,3]))
# plt.scatter(x_standard[y==0,0],x_standard[y==0,1],color="r")
# plt.scatter(x_standard[y==1,0],x_standard[y==1,1],color="g")
# plt.show()