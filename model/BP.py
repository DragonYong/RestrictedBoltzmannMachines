#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time     : 4/20/21-14:55
# @Author   : TuringEmmy
# @Email    : yonglonggeng@163.com
# @WeChat   : superior_god
# @File     : BP.py.py
# @Project  : 00PythonProjects
import tensorflow as tf
from tensorflow.contrib import layers


class BPNeuralNetwork:
    def __init__(self, inpt, n_in, n_out, activation='tanh'):
        """
        神经网络算法构造函数
        :param layers: 神经元层数
        :param activation: 使用的函数（默认tanh函数）
        :return:none
        """
        # if activation == 'logistic':
        #     self.activation = logistic
        #     self.activation_deriv = logistic_derivative
        # elif activation == 'tanh':
        #     self.activation = tanh
        #     self.activation_deriv = tan_deriv

        with tf.name_scope('params'):
            with tf.name_scope('weights'):
                self.W = tf.Variable(
                    tf.random_normal([n_in, n_out], mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None))
                # tf.summary.histogram('weights', self.W)
            # bias
            with tf.name_scope('bias'):
                self.b = tf.Variable(tf.ones([n_out, ]), dtype=tf.float32)

        self.output = tf.matmul(inpt, self.W) + self.b
        tf.add_to_collection('predict', self.output)
        self.params = [self.W, self.b]

    def cost(self, y):
        """
        y: tf.Tensor, the target of the input
        """
        # cross_entropy交叉熵
        with tf.name_scope('loss'):
            # clip_by_value(v,min,max) 截取v,<min表示为min,>max表示为max
            # opt = tf.clip_by_value(self.output, clip_value_min=1e-10, clip_value_max=1.0)
            # softmax_cross_entropy_with_logits(logits,labels,name=None) 相似性概率
            # cost_ = tf.reduce_mean(tf.square(tf.subtract(y ,self.output)))+layers.l2_regularizer(0.01)(self.W)
            cost_ = tf.sqrt(tf.losses.mean_squared_error(y, self.output) + layers.l2_regularizer(0.01)(self.W))
            # summary.scalar对标量数据汇总
            tf.summary.scalar('loss', -cost_)
            return cost_

    def accuarcy(self, y):
        return tf.sqrt(tf.losses.mean_squared_error(y, self.output))
