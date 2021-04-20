#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time     : 4/20/21-13:36
# @Author   : TuringEmmy
# @Email    : yonglonggeng@163.com
# @WeChat   : superior_god
# @File     : predict_dbn.py
# @Project  : 00PythonProjects
import argparse
import os

import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--auther', help='inner batch size', default="TuringEmmy", type=str)
parser.add_argument('--MODEL', help='inner batch size', default=None, type=str)
parser.add_argument('--FEATURE', help='inner batch size', default=None, type=str)

args = parser.parse_args()


def predict(data):
    with tf.Session() as sess:
        meta_path = os.path.join(args.MODEL, "model.ckpt.meta")
        ckpt_path = os.path.join(args.MODEL, "model.ckpt")
        new_saver = tf.train.import_meta_graph(meta_path)
        new_saver.restore(sess, ckpt_path)
        # tf.get_collection() 返回一个list. 但是这里只要第一个参数即可
        y = tf.get_collection('predict')[0]

        graph = tf.get_default_graph()
        # for op in graph.get_operations():
        #     print(op.name, op.values())

        # 因为y中有placeholder，所以sess.run(y)的时候还需要用实际待预测的样本以及相应的参数来填充这些placeholder，而这些需要通过graph的get_operation_by_name方法来获取。
        input_x = graph.get_tensor_by_name("Placeholder:0")
        # keep_prob = graph.get_operation_by_name('keep_prob').outputs[0]
        input = data
        # 使用y进行预测
        res = sess.run(y, feed_dict={input_x: input})
        print(res)
        return res


if __name__ == '__main__':
    data = np.array(args.FEATURE.split(','))

    res = predict(np.expand_dims(data, axis=0))
    print(res)
