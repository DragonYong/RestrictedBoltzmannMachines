#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time     : 4/20/21-14:03
# @Author   : TuringEmmy
# @Email    : yonglonggeng@163.com
# @WeChat   : superior_god
# @File     : model.py
# @Project  : 00PythonProjects
import os
import timeit

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

from model.BP import BPNeuralNetwork
from model.MLP import HiddenLayer

mm = MinMaxScaler()


def MaxMinNormalization(x):
    mm_data = mm.fit_transform(x)
    return mm_data


class RBM(object):
    """A Restricted Boltzmann Machines class"""

    def __init__(self, inpt=None, n_visiable=784, n_hidden=500, W=None,
                 hbias=None, vbias=None):
        """
        :param inpt: Tensor, the input tensor [None, n_visiable]
        :param n_visiable: int, number of visiable units
        :param n_hidden: int, number of hidden units
        :param W, hbias, vbias: Tensor, the parameters of RBM (tf.Variable)
        """
        self.n_visiable = n_visiable
        self.n_hidden = n_hidden
        # Optionally initialize input
        if inpt is None:
            inpt = tf.placeholder(dtype=tf.float32, shape=[None, self.n_visiable])
        self.input = inpt
        # Initialize the parameters if not given
        if W is None:
            # bounds = 4.0 * np.sqrt(6.0 / (self.n_visiable + self.n_hidden))
            bounds = -4.0 * np.sqrt(6.0 / (self.n_visiable + self.n_hidden))
            W = tf.Variable(tf.random_uniform([self.n_visiable, self.n_hidden], minval=-bounds,
                                              maxval=bounds), dtype=tf.float32)
        if hbias is None:
            hbias = tf.Variable(tf.zeros([self.n_hidden, ]), dtype=tf.float32)
        if vbias is None:
            vbias = tf.Variable(tf.zeros([self.n_visiable, ]), dtype=tf.float32)
        self.W = W
        self.hbias = hbias
        self.vbias = vbias
        # keep track of parameters for training (DBN)
        self.params = [self.W, self.hbias, self.vbias]

    def propup(self, v):
        """Compute the sigmoid activation for hidden units given visible units"""
        return tf.nn.sigmoid(tf.matmul(v, self.W) + self.hbias)

    def propdown(self, h):
        """Compute the sigmoid activation for visible units given hidden units"""
        return tf.nn.sigmoid(tf.matmul(h, tf.transpose(self.W)) + self.vbias)

    def sample_prob(self, prob):
        """Do sampling with the given probability (you can use binomial in Theano)"""
        return tf.nn.relu(tf.sign(prob - tf.random_uniform(tf.shape(prob))))

    def sample_h_given_v(self, v0_sample):
        """Sampling the hidden units given visiable sample"""
        h1_mean = self.propup(v0_sample)
        h1_sample = self.sample_prob(h1_mean)
        return (h1_mean, h1_sample)

    def sample_v_given_h(self, h0_sample):
        """Sampling the visiable units given hidden sample"""
        v1_mean = self.propdown(h0_sample)
        v1_sample = self.sample_prob(v1_mean)
        return (v1_mean, v1_sample)

    def gibbs_vhv(self, v0_sample):
        """Implement one step of Gibbs sampling from the visiable state"""
        h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return (h1_mean, h1_sample, v1_mean, v1_sample)

    def gibbs_hvh(self, h0_sample):
        """Implement one step of Gibbs sampling from the hidden state"""
        v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return (v1_mean, v1_sample, h1_mean, h1_sample)

    def free_energy(self, v_sample):
        """Compute the free energy"""
        wx_b = tf.matmul(v_sample, self.W) + self.hbias
        vbias_term = tf.matmul(v_sample, tf.expand_dims(self.vbias, axis=1))
        hidden_term = tf.reduce_sum(tf.log(1.0 + tf.exp(wx_b)), axis=1)
        return -hidden_term - vbias_term

    def get_train_ops(self, learning_rate=0.1, k=1, persistent=None):
        """
        Get the training opts by CD-k
        :params learning_rate: float
        :params k: int, the number of Gibbs step (Note k=1 has been shown work surprisingly well)
        :params persistent: Tensor, PCD-k (TO DO:)
        """
        # Compute the positive phase
        ph_mean, ph_sample = self.sample_h_given_v(self.input)
        # The old state of the chain
        if persistent is None:
            chain_start = ph_sample
        else:
            chain_start = persistent

        # Use tf.while_loop to do the CD-k
        cond = lambda i, nv_mean, nv_sample, nh_mean, nh_sample: i < k
        body = lambda i, nv_mean, nv_sample, nh_mean, nh_sample: (i + 1,) + self.gibbs_hvh(nh_sample)
        i, nv_mean, nv_sample, nh_mean, nh_sample = tf.while_loop(cond, body, loop_vars=[tf.constant(0),
                                                                                         tf.zeros(tf.shape(self.input)),
                                                                                         tf.zeros(tf.shape(self.input)),
                                                                                         tf.zeros(
                                                                                             tf.shape(chain_start)),
                                                                                         chain_start])
        """
        # Compute the update values for each parameter
        update_W = self.W + learning_rate * (tf.matmul(tf.transpose(self.input), ph_mean) - 
                                tf.matmul(tf.transpose(nv_sample), nh_mean)) / tf.to_float(tf.shape(self.input)[0])  # use probability
        update_vbias = self.vbias + learning_rate * (tf.reduce_mean(self.input - nv_sample, axis=0))   # use binary value
        update_hbias = self.hbias + learning_rate * (tf.reduce_mean(ph_mean - nh_mean, axis=0))       # use probability
        # Assign the parameters new values
        new_W = tf.assign(self.W, update_W)
        new_vbias = tf.assign(self.vbias, update_vbias)
        new_hbias = tf.assign(self.hbias, update_hbias)
        """
        chain_end = tf.stop_gradient(nv_sample)  # do not compute the gradients
        cost = tf.reduce_mean(self.free_energy(self.input)) - tf.reduce_mean(self.free_energy(chain_end))
        # Compute the gradients
        gparamm = tf.gradients(ys=[cost], xs=self.params)
        gparams = []
        for i in range(len(gparamm)):
            gparams.append(tf.clip_by_value(gparamm[i], clip_value_min=-1, clip_value_max=1))
        new_params = []
        for gparam, param in zip(gparams, self.params):
            new_params.append(tf.assign(param, param - gparam * learning_rate))

        if persistent is not None:
            new_persistent = [tf.assign(persistent, nh_sample)]
        else:
            new_persistent = []
        return new_params + new_persistent  # use for training

    def get_reconstruction_cost(self):
        """Compute the cross-entropy of the original input and the reconstruction"""
        activation_h = self.propup(self.input)
        activation_v = self.propdown(activation_h)
        # Do this to not get Nan
        activation_v_clip = tf.clip_by_value(activation_v, clip_value_min=1e-10, clip_value_max=1)
        reduce_activation_v_clip = tf.clip_by_value(1.0 - activation_v, clip_value_min=1e-10, clip_value_max=1)
        cross_entropy = -tf.reduce_mean(tf.reduce_sum(
            self.input * (tf.log(activation_v_clip)) + (1.0 - self.input) * (tf.log(reduce_activation_v_clip)), axis=1))
        # cross_entropy = tf.reduce_mean(tf.reduce_sum(self.input * (tf.log(activation_v_clip)) + (1.0 - self.input) * (tf.log(reduce_activation_v_clip)), axis=1))
        # cross_entropy = tf.sqrt(tf.reduce_mean(tf.square(self.input - activation_v)))
        return cross_entropy

    def reconstruct(self, v):
        """Reconstruct the original input by RBM"""
        h = self.propup(v)
        return self.propdown(h)


# 搭建模型
class DBN(object):
    """
    An implement of deep belief network
    The hidden layers are firstly pretrained by RBM, then DBN is treated as a normal
    MLP by adding a output layer.
    """

    def __init__(self, n_in=784, n_out=10, hidden_layers_sizes=[500, 500]):
        """
        :param n_in: int, the dimension of input
        :param n_out: int, the dimension of output
        :param hidden_layers_sizes: list or tuple, the hidden layer sizes
        """
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        # Number of layers
        assert len(hidden_layers_sizes) > 0
        self.n_layers = len(hidden_layers_sizes)
        self.layers = []  # normal sigmoid layer
        self.rbm_layers = []  # RBM layer
        self.params = []  # keep track of params for training

        # Define the input and output
        self.x = tf.placeholder(tf.float32, shape=[None, n_in])
        self.y = tf.placeholder(tf.float32, shape=[None, n_out])

        # Contruct the layers of DBN
        with tf.name_scope('DBN_layer'):
            for i in range(self.n_layers):
                if i == 0:
                    layer_input = self.x
                    input_size = n_in
                else:
                    layer_input = self.layers[i - 1].output
                    input_size = hidden_layers_sizes[i - 1]
                # Sigmoid layer
                with tf.name_scope('internel_layer'):
                    sigmoid_layer = HiddenLayer(input=layer_input, n_in=input_size, n_out=hidden_layers_sizes[i],
                                                activation=tf.nn.sigmoid)
                self.layers.append(sigmoid_layer)
                # Add the parameters for finetuning
                self.params.extend(sigmoid_layer.params)
                # Create the RBM layer
                with tf.name_scope('rbm_layer'):
                    self.rbm_layers.append(RBM(inpt=layer_input, n_visiable=input_size, n_hidden=hidden_layers_sizes[i],
                                               W=sigmoid_layer.W, hbias=sigmoid_layer.b))
            # We use the BP layer as the output layer
            with tf.name_scope('output_layer'):
                self.output_layer = BPNeuralNetwork(inpt=self.layers[-1].output, n_in=hidden_layers_sizes[-1],
                                                    n_out=n_out)
        self.params.extend(self.output_layer.params)
        # The finetuning cost
        with tf.name_scope('output_loss'):
            self.cost = self.output_layer.cost(self.y)
        # The accuracy
        self.accuracy = self.output_layer.accuarcy(self.y)

    def pretrain(self, sess, train_x, batch_size=2, pretraining_epochs=20, lr=0.01, k=1,
                 display_step=10):
        """
        Pretrain the layers (just train the RBM layers)
        :param sess: tf.Session
        :param X_train: the input of the train set (You might modidy this function if you do not use the desgined mnist)
        :param batch_size: int
        :param lr: float
        :param k: int, use CD-k
        :param pretraining_epoch: int
        :param display_step: int
        """
        print('Starting pretraining...\n')
        start_time = timeit.default_timer()
        # Pretrain layer by layer
        for i in range(self.n_layers):
            cost = self.rbm_layers[i].get_reconstruction_cost()
            train_ops = self.rbm_layers[i].get_train_ops(learning_rate=lr, k=k, persistent=None)
            batch_num = int(train_x.shape[0] / batch_size)

            for epoch in range(pretraining_epochs):
                avg_cost = 0.0
                for step in range(batch_num - 1):
                    # 训练
                    x_batch = train_x[step * batch_size:(step + 1) * batch_size]

                    sess.run(train_ops, feed_dict={self.x: x_batch})
                    # 计算cost
                    avg_cost += sess.run(cost, feed_dict={self.x: x_batch, }) / batch_num
                    # print(avg_cost)
                # 输出

                if epoch % display_step == 0:
                    print("\tPretraing layer {0} Epoch {1} cost: {2}".format(i, epoch, avg_cost))

        end_time = timeit.default_timer()
        print("\nThe pretraining process ran for {0} minutes".format((end_time - start_time) / 60))

    def finetuning(self, sess, train_x, train_y, test_x, test_y, training_epochs=20, batch_size=50, lr=0.1,
                   display_step=1):
        """
        Finetuing the network
        """
        accu = []
        accuu = []
        print("\nStart finetuning...\n")
        start_time = timeit.default_timer()
        train_op = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(self.cost)
        batch_num = int(train_x.shape[0] / batch_size)
        # merged = tf.summary.merge_all()
        # writer = tf.summary.FileWriter("logs", sess.graph)
        for epoch in range(training_epochs):
            avg_cost = 0.0
            for step in range(batch_num - 1):
                x_batch = train_x[step * batch_size:(step + 1) * batch_size]
                y_batch = train_y[step * batch_size:(step + 1) * batch_size]
                # 训练
                sess.run(train_op, feed_dict={self.x: x_batch, self.y: y_batch})
                # 计算cost
                avg_cost += sess.run(self.cost, feed_dict={self.x: x_batch, self.y: y_batch}) / batch_num
                # 输出
            if epoch % display_step == 0:
                val_acc = sess.run(self.accuracy, feed_dict={self.x: test_x, self.y: test_y})
                # accu.append(val_acc)
                # accuu.append(avg_cost)

                print("\tEpoch {0} cost: {1} accuracy:{2}".format(epoch, avg_cost, val_acc))

            # result = sess.run(merged, feed_dict={self.x: test_x, self.y: test_y})  # 输出
            # writer.add_summary(result, epoch)
        end_time = timeit.default_timer()
        print("\nThe finetuning process ran for {0} minutes".format((end_time - start_time) / 60))
        # y_aix = np.array(accu)
        # y_aix1=np.array(accuu)
        # x_aix = np.transpose(np.arange(1, 6))
        # plt.plot(x_aix, y_aix,label="predict")
        # plt.plot(x_aix,y_aix1,label="real")
        # plt.savefig("E:\\高若涵计算机毕设\\DBN_predict_performance\\picture\\test_p30_f3.jpg")
        # plt.show()

    def predict(self, sess, x_test=None):
        print("\nStart predict...\n")

        # predict_model = theano.function(
        #     inputs=[self.params],
        #     outputs=self.output_layer.y_pre)
        dbn_y_pre_temp = sess.run(self.output_layer.output, feed_dict={self.x: x_test})
        # print(dbn_y_pre_temp)
        dbn_y_pre = pd.DataFrame(mm.inverse_transform(dbn_y_pre_temp))
        dbn_y_pre.to_csv('NSW_06.csv')
        print("\nPredict over...\n")

    # SVR输出预测结果
    def svr_output(self, sess, test_x, test_y):
        input_svr = sess.run(self.layers[-1].output, feed_dict={self.x: test_x})
        print("\nsvr predict...\n")
        svr = SVR(gamma=0.0005, kernel='rbf', C=15, epsilon=0.008)
        rmse = np.sqrt(-cross_val_score(svr, input_svr, test_y.ravel(), scoring="neg_mean_squared_error", cv=5))
        score = cross_val_predict(svr, input_svr, test_y.ravel(), cv=5)
        print(rmse.mean())
        pd.DataFrame(mm.inverse_transform(score.reshape(-1, 1))).to_csv('../test5.csv')

    # 网格搜索最优参数
    def grid_get(self, sess, model, test_x, test_y, param_grid):
        input_svr = sess.run(self.layers[-1].output, feed_dict={self.x: test_x})
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring="neg_mean_squared_error")
        grid_search.fit(input_svr, test_y)
        print(grid_search.best_params_, np.sqrt(-grid_search.best_score_))
        grid_search.cv_results_['mean_test_score'] = np.sqrt(-grid_search.cv_results_['mean_test_score'])
        print(pd.DataFrame(grid_search.cv_results_)[['params', 'mean_test_score', 'std_test_score']])
