import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader


# Prepare dataset using pandas to handle CSV with headers
class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        # 使用 pandas 读取 CSV 文件
        df = pd.read_csv(filepath)
        xy = df.values.astype(np.float32)  # 转为 numpy 数组，并指定数据类型为 float32
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])  # 输入特征（除了最后一列）
        self.y_data = torch.from_numpy(xy[:, [-1]])  # 目标标签（最后一列）

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


# 数据加载器
dataset = DiabetesDataset('C:/Users/25316/Code/刘二大人/diabetes.csv')
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=0)

# Define model using class
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)

    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))  # 中间层使用 sigmoid 激活函数
        x = torch.sigmoid(self.linear2(x))
        x = self.linear3(x)  # 最后一层不需要激活函数，BCELoss 会自动处理 sigmoid
        return x


model = Model()

# Construct loss and optimizer
criterion = torch.nn.BCEWithLogitsLoss()  # 使用 BCEWithLogitsLoss，自动处理 sigmoid 和交叉熵
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training cycle
if __name__ == '__main__':
    for epoch in range(100000000):  # 训练100个epoch
        for i, data in enumerate(train_loader, 0):  # train_loader 是先 shuffle 后 mini_batch
            inputs, labels = data
            optimizer.zero_grad()  # 清空之前的梯度
            y_pred = model(inputs)  # 进行前向传播
            loss = criterion(y_pred, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新模型参数

            # 每10个batch输出一次损失
            if i % 10 == 0:
                print(f'Epoch [{epoch+1}/100], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
