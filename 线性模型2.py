import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 设定数据
x_data = np.array([1.0, 2.0, 3.0])  # 输入数据 x
y_data = np.array([5.0, 8.0, 11.0])  # 目标值 y, 假设 y = 3x + 2


# 线性模型 y = wx + b
def forward(x, w, b):
    return w * x + b


# 均方误差（MSE）损失函数
def loss(x, y, w, b):
    y_pred = forward(x, w, b)  # 计算预测值
    loss_value = np.mean((y_pred - y) ** 2)  # 计算均方误差
    return loss_value


# 设置 w 和 b 的范围
W = np.arange(0.0, 4.1, 0.1)  # 权重 w 的值域
B = np.arange(0.0, 4.1, 0.1)  # 偏置 b 的值域
w, b = np.meshgrid(W, B)  # 创建网格
# print(w,b)
# 初始化一个矩阵来存储每个 (w, b) 对应的损失
loss_surface = np.zeros_like(w)
# print(loss_surface)
# 计算每个 (w, b) 对应的损失值
for i in range(len(W)):
    for j in range(len(B)):
        # 计算每一对 (w, b) 的损失
        loss_value = loss(x_data, y_data, w[j, i], b[j, i])
        loss_surface[j, i] = loss_value

        # 打印每一对 (w, b) 和对应的损失
        print(f"w = {w[j, i]:.1f}, b = {b[j, i]:.1f} => Loss = {loss_value:.4f}")

# 打印计算损失的平均值
print("\n计算完毕，损失函数的值已计算完毕。")

# 绘制 3D 曲面图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制曲面
ax.plot_surface(w, b, loss_surface, cmap='viridis')

# 设置标签
ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('Loss')

# 显示图形
plt.show()
