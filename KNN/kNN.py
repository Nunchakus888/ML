#!/usr/bin/env python
# coding: utf-8

from numpy import *
import operator


def createDataSet():
    """
    创建数据集和标签
     调用方式
     import kNN
     group, labels = kNN.createDataSet()
    """
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0 (intX, dataSet, labels, k):

    # matrix's data constructor (eg: x.shape --> (2, 3))
    dataSetSize = dataSet.shape[0]

    # tile(sourceData, (row, col))
    diffMat = tile(intX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2

    """
    按维度求和，
    若数据结构是二维：
    axis=0 求0维的和（结果依然是二维结构）
    axis=1 求在1维的和（结果为1维）
    """
    sqlDiatances = sqDiffMat.sum(axis=1)
    distances = sqlDiatances ** 0.5
    sortedDistIndicies = distances.argsort()

    classCount = {}

    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

if __name__ == '__main__':
    createDataSet()
    pass