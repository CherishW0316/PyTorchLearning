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

# PyTorch 之 简单的前馈神经网络

![](https://ptorch.com/uploads/8fe3c043b43e22108ba0b924eb7b8948.png)

从图可看出，其实每一层网络所做的就是 y=W×X+b，只不过W的维数由X和输出维数决定，比如X是10维向量，想要输出的维数，也就是中间层的神经元个数为20，那么W的维数就是20x10，b的维数就是20x1，这样输出的y的维数就为20。

中间层的维数可以自己设计，而最后一层输出的维数就是你的分类数目，比如等会儿要做的MNIST数据集是10个数字的分类，那么最后输出层的神经元就为10。

```
# coding=utf-8

import torch
from torch import nn, optim

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets

batch_size = 32
learning_rate = 1e-2
num_epoches = 50

# 下载训练集 MNIST 手写数字训练集
train_dataset = datasets.MNIST(
    root='./data', train=True, transform=transforms.ToTensor(), download=True)

test_dataset = datasets.MNIST(
    root='./data', train=False, transform=transforms.ToTensor())

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 定义简单的前馈神经网络
class Neuralnetwork(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Neuralnetwork, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

# 上面定义了三层神经网络，输入是28x28，因为图片大小是28x28，中间两个隐藏层大小分别是300和100，最后是个10分类问题，所以输出层为10.
model = Neuralnetwork(28 * 28, 300, 100, 10)
if torch.cuda.is_available():
    model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epoches):
    print('epoch {}'.format(epoch + 1))
    print('*' * 10)
    running_loss = 0.0
    running_acc = 0.0
    for i, data in enumerate(train_loader, 1):
        img, label = data
        img = img.view(img.size(0), -1)
        if torch.cuda.is_available():
            img = Variable(img).cuda()
            label = Variable(label).cuda()
        else:
            img = Variable(img)
            label = Variable(label)
        # 向前传播
        out = model(img)
        loss = criterion(out, label)
        running_loss += loss.data[0] * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        running_acc += num_correct.data[0]
        # 向后传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 300 == 0:
            print('[{}/{}] Loss: {:.6f}, Acc: {:.6f}'.format(
                epoch + 1, num_epoches, running_loss / (batch_size * i),
                running_acc / (batch_size * i)))
    print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(
        epoch + 1, running_loss / (len(train_dataset)), running_acc / (len(
            train_dataset))))
    model.eval()
    eval_loss = 0.
    eval_acc = 0.
    for data in test_loader:
        img, label = data
        img = img.view(img.size(0), -1)
        if torch.cuda.is_available():
            img = Variable(img, volatile=True).cuda()
            label = Variable(label, volatile=True).cuda()
        else:
            img = Variable(img, volatile=True)
            label = Variable(label, volatile=True)
        out = model(img)
        loss = criterion(out, label)
        eval_loss += loss.data[0] * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        eval_acc += num_correct.data[0]
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
        test_dataset)), eval_acc / (len(test_dataset))))
    print()

# 保存模型
torch.save(model.state_dict(), './neural_network.pth')
```

