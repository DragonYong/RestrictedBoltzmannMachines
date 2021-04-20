### Restricted Boltzmann Machines (RBM)相关算子的训练，预测，以及接口发布
===========================
#### 00-项目信息
```
作者：TuringEmmy
时间:2021-04-20 14:02:30
详情：DBN+SVR的模型，简单的使用了波士顿房价数据集进行了实验
```
#### 01-环境依赖
```
ubuntu18.04
python3.7
tensorflow1.14
```
#### 02-部署步骤
##### 训练&验证&测试
```
sh scripts/api.sh  
sh scripts/predict.sh  
sh scripts/train_dbn.sh  
sh scripts/train_rbm.sh
```
##### 线上部署
POST:* Running on http://0.0.0.0:8020/ 
input:
```
{"text":"0.17783, 0.00, 9.690, 0, 0.5850, 5.5690, 73.50, 2.3999, 6, 391.0, 19.20, 395.77, 15.10"}
```
output：
```
{
    "text": [
       0.17783, 0.00, 9.690, 0, 0.5850, 5.5690, 73.50, 2.3999, 6, 391.0, 19.20, 395.77, 15.10
    ],
    "price": [
        [
            -0.8385254144668579
        ]
    ]
}

```

#### 03-目录结构描述
```
.
├── data
│   ├── dbn.py
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── dbn.cpython-36.pyc
│   │   ├── __init__.cpython-36.pyc
│   │   └── rbm.cpython-36.pyc
│   └── rbm.py
├── model
│   ├── BP.py
│   ├── __init__.py
│   ├── MLP.py
│   ├── model.py
│   └── __pycache__
│       ├── BP.cpython-36.pyc
│       ├── __init__.cpython-36.pyc
│       ├── MLP.cpython-36.pyc
│       └── model.cpython-36.pyc
├── predict_dbn.py
├── __pycache__
│   └── predict_dbn.cpython-36.pyc
├── README.md
├── scripts
│   ├── api.sh
│   ├── predict.sh
│   ├── train_dbn.sh
│   └── train_rbm.sh
├── server.py
├── train_dbn.py
└── train_rbm.py
```


#### 04-版本更新
##### V1.0.0 版本内容更新-2021-04-20 15:52:34
- rbm&dbm相关模型训练测试
- dbm离线预测
- dbm离线服务器部署

#### 05-TUDO
- rbm离线预测
- rbm服务器部署