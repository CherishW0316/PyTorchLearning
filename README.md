# PyTorchLearning
从零开始学PyTorch

# 什么是PyTorch
[什么是PyTorch](https://ptorch.com/news/1.html)

# 安装Pytorch
1. 登录[http://pytorch.org/](http://pytorch.org/)，根据自己的需要选择相应的安装方法，注意以sudo执行。
> pip 安装 `sudo easy_install pip`。
如果系统自带numpy版本太低的话，运行`sudo easy_install -U numpy`

# PyTorch 之 神经网络模型建立
模型的建立主要依赖于torch.nn，torch.nn包含这个所有神经网络的层的结构
```
# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

# 基本的网络构建类模版
class net_name(nn.Module):
    def __init__(self):
        super(net_name,self).__init__()
        # 可以添加子网络
        self.conv1 = nn.Conv2d(3,10,3)
        # 具体每种层的参数可以查看文档

    def forward(self, x):
        # 定向前传播
        out = self.conv1(x)
        return out
```
这就是构建所有神经网络的模板，不管你想构建卷积神经网络还是循环神经网络或者是生成对抗网络都依赖于这个结构

# PyTorch 之 线性回归
![](https://ptorch.com/uploads/e7d5687f7ffd9f4d67e169098cd3f66e.png)
1. 为方便绘图，先安装`sudo pip install matplotlib`
2. 建立文件linear.py
```
import numpy as np
import matplotlib.pyplot as plt

x = np.array([1.3,2.4,3.1,4.32,5.13,6.15,7.25,8.92,8.92,3.12,4.32,8.52], dtype=np.float32)

y = np.array([2.3,1.13,5.4,6.1,6.2,6.85,7.03,8.18,7.25,6.82,1.75,6.2], dtype=np.float32)

plt.scatter(x,y,color="red")
plt.show()
```
运行后会有点状图出现
