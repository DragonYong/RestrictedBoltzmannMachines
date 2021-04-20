#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time     : 4/20/21-14:32
# @Author   : TuringEmmy
# @Email    : yonglonggeng@163.com
# @WeChat   : superior_god
# @File     : train_rbm.py
# @Project  : 00PythonProjects
import argparse
import os

import numpy as np
import tensorflow as tf

from data.rbm import loadDataSet, splitDataSet
from model.model import RBM

parser = argparse.ArgumentParser()
parser.add_argument('--auther', help='inner batch size', default="TuringEmmy", type=str)
parser.add_argument('--MODEL', help='inner batch size', default=None, type=str)
parser.add_argument('--DATASET', help='inner batch size', default=None, type=str)
parser.add_argument('--EPOCH', help='inner batch size', default=1000, type=int)
parser.add_argument('--RATIO', help='inner batch size', default=0.9, type=float)
parser.add_argument('--LEARNING_RATE', help='inner batch size', default=0.00001, type=float)

args = parser.parse_args()

trainSet, testSet = loadDataSet(args.DATASET, 0.9)
trainX, trainY = splitDataSet(trainSet)
testX, testY = splitDataSet(testSet)
# print(testX.shape[0])
# print(testX)
xs = tf.placeholder(dtype=tf.float32, shape=[None, 13])
mean, std = tf.nn.moments(xs, axes=[0])
scale = 0.1
shift = 0
epsilon = 0.001
data = tf.nn.batch_normalization(xs, mean, std, shift, scale, epsilon)
x = tf.placeholder(dtype=tf.float32, shape=[None, 13])

tf.set_random_seed(seed=99999)
np.random.seed(123)

rbm = RBM(inpt=x, n_visiable=13, n_hidden=2)
train_ops = rbm.get_train_ops(learning_rate=args.LEARNING_RATE, k=1, persistent=None)
cost = rbm.get_reconstruction_cost()
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    sess.run(mean, feed_dict={xs: trainX})
    sess.run(std, feed_dict={xs: trainX})
    num = sess.run(data, feed_dict={xs: trainX})
    # print()
    # print(rbm.W.eval())
    saver = tf.train.Saver()
    for epoch in range(args.EPOCH):
        avg_cost = 0
        sess.run(train_ops, feed_dict={x: num})
        # 计算cost
        avg_cost += sess.run(cost, feed_dict={x: num, })
        # print >> sys.stderr, 'Training epoch %d, cost is ' % epoch, cost
        print(avg_cost)

        if epoch % 100 == 0:
            ckpt_path = os.path.join(args.MODEL, "model.ckpt")  # 将保存到当前目录下的的saved_model文件夹下model.ckpt文件
            # file_name = 'saved_model/model.ckpt'
            saver.save(sess, ckpt_path)
