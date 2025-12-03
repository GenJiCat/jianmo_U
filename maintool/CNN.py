import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ======================
# 1. 数据准备
# ======================
transform = transforms.Compose([
    transforms.ToTensor(),   # 转换为Tensor
    transforms.Normalize((0.5,), (0.5,))  # 标准化 [-1, 1]
])

train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# ======================
# 2. 定义CNN模型
# ======================
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 输入: 1×28×28
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1) # 输出: 32×28×28
        self.pool = nn.MaxPool2d(2, 2)                                   # 输出: 32×14×14
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) # 输出: 64×14×14
        # 池化后输出: 64×7×7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10) # 分类10类

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 卷积+ReLU+池化
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # 展平
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x  # 注意这里没加Softmax，因为CrossEntropyLoss自带

# ======================
# 3. 训练模型
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 2

for epoch in range(epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # 前向
        output = model(data)
        loss = criterion(output, target)

        # 反向
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx}], Loss: {loss.item():.4f}")

# ======================
# 4. 测试模型
# ======================
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")
