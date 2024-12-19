import torch

# 数据
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])

# 定义线性模型
class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)  # 输入和输出都是1维

    def forward(self, x):
        return self.linear(x)

model = LinearModel()

# 损失函数和优化器
criterion = torch.nn.MSELoss()  # 均方误差损失
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 随机梯度下降，学习率0.01

# 训练过程
for epoch in range(1000):
    y_pred = model(x_data)  # 前向传播
    loss = criterion(y_pred, y_data)  # 计算损失
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')  # 打印损失

    optimizer.zero_grad()  # 梯度清零
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数

# 打印模型参数
print(f'Weight (w) = {model.linear.weight.item()}')
print(f'Bias (b) = {model.linear.bias.item()}')

# 测试
x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print(f'y_pred = {y_test.data.item()}')
