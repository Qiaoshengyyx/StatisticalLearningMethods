# -*- coding: utf-8 -*-
# @Author  : YuanYuxuan
# @FileName: chapter3.py
# @Software: PyCharm
from copy import copy, deepcopy

from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from collections import Counter
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

def cal_euclid_dis(v1, v2):
    return np.sqrt(np.sum((v1-v2)**2))


class KNearestNeighbors(object):
    def __init__(self, X, y, k=1, disfunc=cal_euclid_dis):
        self.X = X
        self.y = y
        self.k = k
        self.disfunc = disfunc
        pass


    def fit(self, X, y, disfunc=cal_euclid_dis):
        pass

    def predict(self, x):
        # 预测单个样本
        dist = list(map(lambda v: self.disfunc(v, x), self.X))
        kneighbors = np.argsort(dist)[:self.k]
        # 确定类别
        pred_labels = [self.y[i] for i in kneighbors]
        cnt = Counter(pred_labels)
        pred = cnt.most_common()[0][0]
        return pred


class Node(object):
    def __init__(self, vec, j):
        self.vec = vec
        self.j = j
        self.left = None
        self.right = None



class KDTree(object):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        if isinstance(X, list):
            self.X = np.array(X)
        self.k = self.X.shape[1]
        self.root = self.fit(self.X, 0)


    def fit(self, data:np.array, j):
        '''
        构造kd树
        :param X: np.array 数据集
        :param j: 当前深度
        :return: kd树根结点
        '''
        if data.shape[0] == 0:
            return None

        # 用于切分的维度
        l = j%self.k
        # 中位数位置
        mid = data.shape[0] // 2
        # 排序
        sorted_data = data[np.argsort(data[:, l])]
        root = Node(sorted_data[mid], l)

        root.left = self.fit(sorted_data[ :mid], j+1)
        root.right = self.fit(sorted_data[mid+1: ], j+1)
        return root

    def search(self, node:Node, x, min_dis:list):
        '''
        :param x: 输入
        :param min_dis: 当前最短距离, 当前最近邻点
        :return: 最近邻点
        '''
        if node is None:
            return
        # 先找到距离最近的叶子结点
        dis = cal_euclid_dis(x, node.vec)
        # 不断更新当前最近邻点
        if dis < min_dis[0] or min_dis[0] < 0:
            min_dis[0] = dis
            min_dis[1] = node.vec.tolist()
        # 根据不同维度与当前根结点的比较判断进入左右子树
        # 找到第一个目标叶子结点作为当前最近邻点之后，

        flag = 0
        if x[node.j] <= node.vec[node.j]:
            self.search(node.left, x, min_dis)
        else:
            self.search(node.right, x, min_dis)
            flag = 1

        # 检查另一子结点对应的区域是否与以目标点为球心、以当前最短距离为半径的超球体相交
        if flag:
            if node.left and cal_euclid_dis(node.left.vec, x) < min_dis[0]:
                self.search(node.left, x, min_dis)
        else:
            if node.right and cal_euclid_dis(node.right.vec, x) < min_dis[0]:
                self.search(node.right, x, min_dis)



if __name__ == '__main__':
    # knn
    # iris = datasets.load_iris()
    # X = iris['data']
    # y = iris['target']
    # train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=3)
    # knn = KNearestNeighbors(train_x, train_y , k=5, disfunc=cal_euclid_dis)
    # pred_y = [knn.predict(x) for x in test_x]
    # print(classification_report(test_y, pred_y))

    # 习题3.1
    # 样例点
    # X = np.array(
    #     [[0.5, 0.9], [0.7, 2.8], [1.3, 4.6], [1.4, 2.8], [1.7, 0.8], [1.9, 1.4], [2, 4], [2.3, 3], [2.5, 2.5], [2.9, 2],
    #      [2.9, 3], [3, 4.5], [3.3, 1.1], [4, 3.7], [4, 2.2], [4.5, 2.5], [4.6, 1], [5, 4]])
    # y = np.array([0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0])
    #
    # # 设置二维网格的边界
    # x_min, x_max = 0, 6
    # y_min, y_max = 0, 6
    #
    # # 设置不同类别区域的颜色
    # cmap_light = ListedColormap(['#FFFFFF', '#BDBDBD'])
    # # 为了给区域上色，生成那么多点，实际是网格点，间隔0.01
    # h = 0.01
    # _x, _y = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    #
    # # k = 1 or 2
    # knn = KNeighborsClassifier(n_neighbors=1)
    # knn.fit(X, y)
    #
    # pred = knn.predict(np.c_[_x.ravel(), _y.ravel()])
    # pred = pred.reshape(_x.shape)
    #
    # plt.figure()
    # plt.xticks([i for i in range(7)])
    # plt.yticks([i for i in range(7) if i != 0])
    # # plt.xticks(tuple([x for x in range(6)]))
    # # plt.yticks(tuple([y for y in range(6) if y != 0]))
    #
    # plt.pcolormesh(_x, _y, pred, cmap=cmap_light)
    #
    # # 设置坐标轴标签
    # plt.xlabel('x')
    # plt.ylabel('y')

    # 绘制实例点的散点图
    # plt.scatter(X[:, 0], X[:, 1], c=y)
    # plt.show()


    # kd树测试
    X = np.array([[2, 3],
                  [5, 4],
                  [9, 6],
                  [4, 7],
                  [8, 1],
                  [7, 2]])
    y = None
    test_x = np.array([3, 4.5])
    min_dis = [-1.0, []]

    kdtree = KDTree(X, y)
    kdtree.search(kdtree.root, test_x, min_dis)
    print("最短距离：{} \n最近邻点：{}".format(min_dis[0], min_dis[1]))