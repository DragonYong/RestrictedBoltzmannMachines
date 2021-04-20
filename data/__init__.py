#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time     : 4/20/21-14:25
# @Author   : TuringEmmy
# @Email    : yonglonggeng@163.com
# @WeChat   : superior_god
# @File     : __init__.py.py
# @Project  : 00PythonProjects
from sklearn.preprocessing import MinMaxScaler

mm = MinMaxScaler()


def MaxMinNormalization(x):
    mm_data = mm.fit_transform(x)
    return mm_data
