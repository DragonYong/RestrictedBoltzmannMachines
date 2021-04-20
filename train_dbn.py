#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time     : 4/20/21-13:43
# @Author   : TuringEmmy
# @Email    : yonglonggeng@163.com
# @WeChat   : superior_god
# @File     : train_dbn.py
# @Project  : 00PythonProjects
import argparse
import os
import timeit

import numpy as np
import pandas as pd
import tensorflow as tf

from data.dbn import splitDataSet

from model.model import DBN, MaxMinNormalization

parser = argparse.ArgumentParser()
parser.add_argument('--auther', help='inner batch size', default="TuringEmmy", type=str)
parser.add_argument('--MODEL', help='模型位置', default="TuringEmmy", type=str)
parser.add_argument('--DATASET', help='inner batch size', default="TuringEmmy", type=str)
parser.add_argument('--EPOCH', help='训练次数', default=200, type=int)
parser.add_argument('--BATCH_SIZE', help='inner batch size', default=30, type=int)
parser.add_argument('--LEARNING_RATE', help='学习率', default=0.0001, type=float)
parser.add_argument('--DISPLAY_STEP', help='多少个epoch打印一侧结果', default=10, type=int)

args = parser.parse_args()


def train():
    graph = tf.Graph()
    with graph.as_default():
        input_size = 13  # 你输入的数据特征数量
        start_time0 = timeit.default_timer()
        dataMat, labelMat = splitDataSet(args.DATASET)
        # print(dataMat[0:3])
        # print(labelMat[0:3])
        trainX = dataMat[:300, :]
        trainY = labelMat[:300]
        testX = dataMat[300:, :]
        testY = labelMat[300:]
        pd.DataFrame(testY).to_csv('testY.csv')

        x_train = MaxMinNormalization(trainX)
        # print(x_train[0:3])
        y_train = MaxMinNormalization(np.transpose([trainY]))
        # print(y_train[0:3])
        x_test = MaxMinNormalization(testX)

        y_test = MaxMinNormalization(np.transpose([testY]))

        sess = tf.Session(graph=graph)
        dbn = DBN(n_in=x_train.shape[1], n_out=1, hidden_layers_sizes=[13, 15])
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()
        tf.set_random_seed(seed=1111)

        dbn.pretrain(sess, x_train, lr=args.LEARNING_RATE, pretraining_epochs=100, batch_size=args.BATCH_SIZE,
                     display_step=args.DISPLAY_STEP)
        dbn.finetuning(sess, x_train, y_train, x_test, y_test, lr=args.LEARNING_RATE, training_epochs=args.EPOCH,
                       batch_size=args.BATCH_SIZE, display_step=args.DISPLAY_STEP)
        # dbn.grid_get(sess,model=SVR(),test_x=x_test,test_y=y_test,param_grid={'C': [9, 11, 13, 15], 'kernel': ["rbf"], "gamma": [0.0003, 0.0004,0.0005], "epsilon": [0.008, 0.009]})
        dbn.svr_output(sess, x_test, y_test)
        # dbn.predict(sess, x_test)

        ckpt_path = os.path.join(args.MODEL, "model.ckpt")  # 将保存到当前目录下的的saved_model文件夹下model.ckpt文件
        # file_name = 'saved_model/model.ckpt'
        saver.save(sess, ckpt_path)

        end_time0 = timeit.default_timer()
        print("\nThe Predict process ran for {0}".format((end_time0 - start_time0)))


train()
