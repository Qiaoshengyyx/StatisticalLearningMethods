# -*- coding: utf-8 -*-
# @Author  : YuanYuxuan
# @FileName: chapter2.py
# @Software: PyCharm

from sklearn import datasets
import numpy as np


class Perceptron(object):
    def __init__(self, X, y, learning_rate=1):
        '''
        :param X: 特征向量集合, array
        :param y: 标签集合, array
        '''
        self.X = X
        self.y = y
        if isinstance(self.X, list):
            self.X = np.array(self.X)
        if isinstance(self.y, list):
            self.y = np.array(self.y)
        self.n, self.m = self.X.shape[0], self.X.shape[1] # 样本数量 特征向量维度
        # w与b初始化
        self.w = np.zeros(self.m)
        self.b = 0
        self.eta = learning_rate
        self.alpha = np.zeros(self.n)

        self.t = 0  # 迭代次数
        self.Gram = None
        pass

    def sign(self, x):
        if x == 0:
            return 0
        return 1 if x >= 0 else -1

    def fit_original(self):
        # 学习算法的原始形式
        print('迭代次数:0\t误分类点: \tw:0\tb:0\n')
        self.t = 0
        while True:
            i=0
            while i<self.n:
                if not self.judge_origin(self.X[i], self.y[i]):
                    self.update_origin(self.X[i], self.y[i])
                    print('迭代次数:{}\t误分类点:{}\tw:{}\tb:{}\n'.format(self.t, i + 1, self.w, self.b))
                    break
                i+=1
            if i == self.n:
                self.t+=1
                print('迭代次数:{}\t误分类点:0\tw:{}\tb:{}\n'.format(self.t, self.w, self.b))
                break



    def fit_duality(self):
        # 学习算法的对偶形式
        # 计算gram矩阵
        self.Gram = self.X.dot(self.X.T)
        self.t = 0
        print('迭代次数:0\t误分类点: \nalpha向量:{}\t b:{}\n'.format(self.alpha, self.b))
        while True:
            i=0
            while i < self.n:
                if not self.judge_duality(i):
                    self.update_duality(i)
                    print('迭代次数：{}\t误分类点:{}\nalpha向量:{}\t b:{}\n'.format(self.t, i+1, self.alpha, self.b))
                    break
                i+=1
            if i == self.n:
                self.t += 1
                print('迭代次数：{}\t误分类点:0 \nalpha向量:{}\t b:{}\n'.format(self.t, self.alpha, self.b))
                break


    def judge_origin(self, x1, y1):
        tmp = self.w.dot(x1)+self.b
        y_pred = self.sign(tmp)
        if y_pred == 0:
            return False
        return True if y_pred==y1 else False

    def update_origin(self, x1, y1):
        # 更新w和b , x1是向量
        self.t += 1
        self.w += self.eta*x1*y1
        self.b += self.eta*y1

    def judge_duality(self, i):
        tmp_sum = np.dot(np.multiply(self.alpha, self.y), self.Gram[i, :])
        if self.y[i]*(tmp_sum + self.b) <= 0:
            return False
        return True

    def update_duality(self, i):
        self.alpha[i] += 1
        self.b += self.y[i]
        self.t += 1

if __name__ == '__main__':
    X = [[3,3],[4,3],[1,1]]
    y = [1,1,-1]
    p = Perceptron(X, y, learning_rate=1)
    p.fit_original()
    p.fit_duality()