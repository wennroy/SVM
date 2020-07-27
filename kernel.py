# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 14:14:37 2020

@author: lengwaifang
"""


import numpy as np
from matplotlib import pyplot as plt

def load_data(filename):
    dataset, labels = [], []
    with open(filename, 'r') as f:
        for line in f:
            x, y, label = [float(i) for i in line.strip().split()]
            dataset.append([x, y])
            labels.append(label)
    return dataset, labels
def clip(alpha, L, H):
    ''' 修建alpha的值到L和H之间.
    '''
    if alpha < L:
        return L
    elif alpha > H:
        return H
    else:
        return alpha
def select_j(i, m):
    ''' 在m中随机选择除了i之外剩余的数
    '''
    l = list(range(m))
    seq = l[: i] + l[i+1:]
    return random.choice(seq)

def get_w(alphas, dataset, labels):
    ''' 通过已知数据点和拉格朗日乘子获得分割超平面参数w
    '''
    alphas, dataset, labels = np.array(alphas), np.array(dataset), np.array(labels)
    yx = labels.reshape(1, -1).T*phi(dataset)
    w = np.dot(yx.T, alphas)
    return w.tolist()

def clip(alpha, L, H):
    ''' 修建alpha的值到L和H之间.
    '''
    if alpha < L:
        return L
    elif alpha > H:
        return H
    else:
        return alpha
def select_j(i, m):
    ''' 在m中随机选择除了i之外剩余的数
    '''
    l = list(range(m))
    seq = l[: i] + l[i+1:]
    return np.random.choice(seq)

def K(x,y):
    '''
    输入x为向量，自定义核函数。
    '''
    x = np.array(x)
    y = np.array(y)
    return (np.matmul(x,y) + 1)**2
def phi(x):
    '''
    需输入[[1,2],[2,3]]这样
    '''
    n = len(x)
    x = np.array(x).reshape(2,-1)
    return np.vstack([x[0]**2,x[1]**2,np.sqrt(2)*x[0],np.sqrt(2)*x[1],np.sqrt(2)*x[0]*x[1],np.ones(n)]).T
 
def simple_smo(dataset, labels, C, max_iter):
    ''' 
    简化版SMO算法实现，未使用启发式方法对alpha对进行选择.
    :param dataset: 所有特征数据向量
    :param labels: 所有的数据标签
    :param C: 软间隔常数, 0 <= alpha_i <= C
    :param max_iter: 外层循环最大迭代次数
    '''
    dataset = np.array(dataset)
    m, n = dataset.shape
    labels = np.array(labels)
    # 初始化参数
    alphas = np.zeros(m)
    b = 0
    it = 0
    def f(x):
        "SVM分类器函数 y = w^Tx + b"
        # Kernel function vector.
        x = np.matrix(x)
        w = get_w(alphas,dataset,labels)
        fx = np.matmul(phi(x),w) + b
        return fx[0]

    while it < max_iter:
        pair_changed = 0
        for i in range(m):
            a_i, x_i, y_i = alphas[i], dataset[i], labels[i]
            fx_i = f(x_i)
            E_i = fx_i - y_i
            j = select_j(i, m)
            a_j, x_j, y_j = alphas[j], dataset[j], labels[j]
            fx_j = f(x_j)
            E_j = fx_j - y_j
            K_ii, K_jj, K_ij = K(x_i, x_i), K(x_j, x_j), K(x_i, x_j)
            eta = K_ii + K_jj - 2*K_ij
            if eta <= 0:
                print('WARNING  eta <= 0')
                continue
            # 获取更新的alpha对
            a_i_old, a_j_old = a_i, a_j
            a_j_new = a_j_old + y_j*(E_i - E_j)/eta
            # 对alpha进行修剪
            if y_i != y_j:
                L = max(0, a_j_old - a_i_old)
                H = min(C, C + a_j_old - a_i_old)
            else:
                L = max(0, a_i_old + a_j_old - C)
                H = min(C, a_j_old + a_i_old)
            a_j_new = clip(a_j_new, L, H)
            a_i_new = a_i_old + y_i*y_j*(a_j_old - a_j_new)
            if abs(a_j_new - a_j_old) < 0.00001:
                #print('WARNING   alpha_j not moving enough')
                continue
            alphas[i], alphas[j] = a_i_new, a_j_new
            # 更新阈值b
            b_i = -E_i - y_i*K_ii*(a_i_new - a_i_old) - y_j*K_ij*(a_j_new - a_j_old) + b
            b_j = -E_j - y_i*K_ij*(a_i_new - a_i_old) - y_j*K_jj*(a_j_new - a_j_old) + b
            if 0 < a_i_new < C:
                b = b_i
            elif 0 < a_j_new < C:
                b = b_j
            else:
                b = (b_i + b_j)/2
            pair_changed += 1
            # print('INFO   iteration:{}  i:{}  pair_changed:{}'.format(it, i, pair_changed))
        if pair_changed == 0:
            it += 1
        else:
            it = 0
        print('iteration number: {}'.format(it))
    return alphas, b


if '__main__' == __name__:
    # 加载训练数据
    # from sklearn import  datasets
    # dataset = np.array([[4,3],
    #           [5,4],
    #           [4,5],
    #           [1,1],
    #           [2,4],
    #           [3,5],
    #           [4,2],
    #           [1,2],
    #           [2,1],
    #           [3,2]])
    # labels = np.array([1,1,1,-1,-1,1,-1,-1,-1,-1])
    # iris = datasets.load_iris()
    # dataset = iris.data
    # labels = iris.target
    
    
    from basedata1 import datasetobtain
    dataset, labels = datasetobtain()
    
    
    # 使用简化版SMO算法优化SVM
    alphas, b = simple_smo(dataset, labels,100, 30)
    # 分类数据点
    classified_pts = {'+1': [], '-1': []}
    for point, label in zip(dataset, labels):
        if label == 1.0:
            classified_pts['+1'].append(point)
        else:
            classified_pts['-1'].append(point)
    w = get_w(alphas, dataset, labels)
    w = [w[1],w[2],w[3]*np.sqrt(2),w[4]*np.sqrt(2),w[5]*np.sqrt(2),w[0] + b]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    from draw_eclipse import parameter_eclipse
    mu1, mu2, width, height, angle = parameter_eclipse(w)
    
    from matplotlib.patches import Ellipse
    e = Ellipse(xy = (mu1,mu2), height = height*2, width =width * 2, angle=angle)
    ax.add_artist(e)
    e.set_facecolor("white")
    # 绘制数据点
    for label, pts in classified_pts.items():
        pts = np.array(pts)
        ax.scatter(pts[:, 0], pts[:, 1], label=label)
    # 绘制分割线
    x1, _ = max(dataset, key=lambda x: x[0])
    x2, _ = min(dataset, key=lambda x: x[0])
    # 绘制支持向量
    for i, alpha in enumerate(alphas):
        if abs(alpha) > 1e-3:
            x, y = dataset[i]
            ax.scatter([x], [y], s=150, c='none', alpha=0.7,
                       linewidth=1.5, edgecolor='#AB3319')
    plt.show()
    
# https://zhuanlan.zhihu.com/p/29212107