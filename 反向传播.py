import numpy as np
import matplotlib.pyplot as plt
import torch

x_data = [1.0,2.0,3.0]
y_data = [2.0,4.0,6.0]

w = torch.Tensor([1.0])#初始权值
w.requires_grad = True#计算梯度，默认是不计算的

def forward(x):
    return x * w

def loss(x,y):#构建计算图
    y_pred = forward(x)
    return (y_pred-y) **2  # 损失函数

print('Predict (befortraining)',4,forward(4))

for epoch in range(100):
    l = loss(1, 2)#为了在for循环之前定义l,以便之后的输出，无实际意义
    for x,y in zip(x_data,y_data):
        l = loss(x, y)
        l.backward()
        print('\tgrad:',x,y,w.grad.item())
        w.data = w.data - 0.01*w.grad.data #注意这里的grad是一个tensor，所以要取他的data
        w.grad.data.zero_() #释放之前计算的梯度
    print('Epoch:',epoch,l.item())

print('Predict(after training)',4,forward(4).item())

