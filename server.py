#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time     : 4/20/21-15:22
# @Author   : TuringEmmy
# @Email    : yonglonggeng@163.com
# @WeChat   : superior_god
# @File     : api.sh.py
# @Project  : 00PythonProjects
import argparse
import json
import os

import numpy as np
from flask import Flask, request

from predict_dbn import predict

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

app = Flask(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--auther', help='inner batch size', default="TuringEmmy", type=str)
parser.add_argument('--MODEL', help='Model path', default=None, type=str)
args = parser.parse_args()
print(args.__dict__)


@app.route('/do', methods=['POST'])
def event_extraction():
    data = request.data
    data_str = np.array(eval(data)['text'].split(','))
    data_str = [float(x) for x in data_str]
    feature_matrix = np.expand_dims(data_str, axis=0)
    price = predict(feature_matrix)
    # print(type(price),str(price))
    res = {"text": data_str, "price": price.tolist()}
    return json.dumps(res, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8020)
