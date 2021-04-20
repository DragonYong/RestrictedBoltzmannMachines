#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time     : 4/20/21-14:27
# @Author   : TuringEmmy
# @Email    : yonglonggeng@163.com
# @WeChat   : superior_god
# @File     : rbm.py
# @Project  : 00PythonProjects
import random

import numpy as np


def loadDataSet(fileName, ratio):
    trainingData = []
    testData = []
    with open(fileName) as txtData:
        lines = txtData.readlines()
        for line in lines:
            lineData = line.strip().split()  # 去除空格
            if random.random() < ratio:  # 数据集分割比例
                trainingData.append(lineData)  # 训练数据集列表
            else:
                testData.append(lineData)  # 测试数据集列表
    return trainingData, testData


def splitDataSet(dataset):
    numFeat = len(dataset[0]) - 1
    dataMat = []
    labelMat = []
    x = np.array(dataset).shape[0]
    for a in range(0, x):
        lineArr = []
        # curLine=dataset[a].strip().split()
        for i in range(numFeat):
            lineArr.append(float(dataset[a][i]))
        dataMat.append(lineArr)
        labelMat.append(float(dataset[a][-1]))
    return np.array(dataMat), np.array(labelMat)
