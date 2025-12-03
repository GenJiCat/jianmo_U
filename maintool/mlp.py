import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# 1. 加载鸢尾花数据集并预处理数据
iris = datasets.load_iris()
X = iris.data  # 形状 (150, 4)
y = iris.target  # 形状 (150,)

# 对特征进行标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 转为 tensor
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# 划分训练集和测试集（80%训练，20%测试）
X_train, X_test, y_train, y_test = train_test_split(
    X_tensor, y_tensor, test_size=0.2, random_state=42
)

# 将处理后的数据保存到 txt 文件中
np.savetxt('X_train.txt', X_train.numpy(), fmt='%.5f')
np.savetxt('y_train.txt', y_train.numpy(), fmt='%d')
np.savetxt('X_test.txt', X_test.numpy(), fmt='%.5f')
np.savetxt('y_test.txt', y_test.numpy(), fmt='%d')


# 2. 定义 MLP 模型：输入层4个神经元，第一隐藏层4个神经元，第二隐藏层3个神经元，输出层3个神经元
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # 输入层到第一隐藏层：输入 4, 输出 4
        self.fc1 = nn.Linear(4, 4)
        # 第一隐藏层到第二隐藏层：输入 4, 输出 3
        self.fc2 = nn.Linear(4, 3)
        # 第二隐藏层到输出层：输入 3, 输出 3
        self.fc3 = nn.Linear(3, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # 此处不加激活函数，后面由 CrossEntropyLoss 内部处理（softmax）
        return x


model = MLP()
print(model)

# 3. 定义训练参数和损失函数
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
epochs = 150

# 4. 训练模型
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# 5. 测试模型并保存正确预测与错误预测的结果
model.eval()
with torch.no_grad():
    outputs_test = model(X_test)
    _, predicted = torch.max(outputs_test, 1)

    # 与真实标签对比
    correct_mask = (predicted == y_test)
    correct_preds = predicted[correct_mask].cpu().numpy()
    incorrect_preds = predicted[~correct_mask].cpu().numpy()



