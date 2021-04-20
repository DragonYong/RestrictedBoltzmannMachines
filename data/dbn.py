#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time     : 4/20/21-14:25
# @Author   : TuringEmmy
# @Email    : yonglonggeng@163.com
# @WeChat   : superior_god
# @File     : dbn.py
# @Project  : 00PythonProjects
import random

import numpy as np


def splitDataSet(filename):
    numFeat = len(open(filename).readline().split()) - 1
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = []
        curline = line.strip().split()
        for i in range(numFeat):
            lineArr.append(float(curline[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curline[-1]))
    return np.array(dataMat), np.array(labelMat)


def loadDataSet(dataset, ratio):
    trainingData = []
    testData = []
    x = np.array(dataset).shape[0]

    for i in range(0, x):
        if random.random() < ratio:  # 数据集分割比例
            trainingData.append(dataset[i])  # 训练数据集列表
        else:
            testData.append(dataset[i])  # 测试数据集列表
    return trainingData, testData
