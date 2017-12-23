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
# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt

x = np.array([1.3,2.4,3.1,4.32,5.13,6.15,7.25,8.92,8.92,3.12,4.32,8.52], dtype=np.float32)

y = np.array([2.3,1.13,5.4,6.1,6.2,6.85,7.03,8.18,7.25,6.82,1.75,6.2], dtype=np.float32)

plt.scatter(x,y,color="red")
plt.show()
```
运行后会有点状图出现
![](https://github.com/CherishW0316/PyTorchLearning/blob/master/image/figure_1.png?raw=true)
3. 
```
x = torch.from_numpy(x)
y = torch.from_numpy(y)
```
将数据从numpy转为tensor

4. 结合基本的网络模板，形成以下代码
```
# coding=utf-8

import torch
from torch import nn, optim
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

# 由numpy转为Tensor
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)


# Linear Regression Model
# 形式同基本模板
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        
        # 这里的nn.Linear表示的是 y=w*x+b，里面的两个参数都是1，表示的是x是1维，y也是1维，当然这里是可以根据你想要的输入输出维度来更改的。
        self.linear = nn.Linear(1, 1)  # input and output is 1 dimension
        
    def forward(self, x):
        out = self.linear(x)
        return out


model = LinearRegression()
# 定义loss和优化函数
criterion = nn.MSELoss() # 最小二乘
optimizer = optim.SGD(model.parameters(), lr=1e-4) # 随机梯度下降，注意需要将model的参数model.parameters()传进去让这个函数知道他要优化的参数是哪些。

# 开始训练
# 第一个循环表示每个epoch，接着开始前向传播，然后计算loss，然后反向传播，接着优化参数，特别注意的是在每次反向传播的时候需要将参数的梯度归零，即optimizer.zero_grad()
num_epochs = 1000
for epoch in range(num_epochs):
    inputs = Variable(x_train)
    target = Variable(y_train)

    # forward
    out = model(inputs)
    loss = criterion(out, target)
    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 20 == 0:
        print('Epoch[{}/{}], loss: {:.6f}'
              .format(epoch+1, num_epochs, loss.data[0]))
              
# 需要用 model.eval()，让model变成测试模式
model.eval()
predict = model(Variable(x_train))
predict = predict.data.numpy()
plt.plot(x_train.numpy(), y_train.numpy(), 'ro', label='Original data')
plt.plot(x_train.numpy(), predict, label='Fitting Line')
# 显示图例
plt.legend() 
plt.show()

# 保存模型
torch.save(model.state_dict(), './linear.pth')
```
运行得到
![](https://github.com/CherishW0316/PyTorchLearning/blob/master/image/image.png?raw=true)
