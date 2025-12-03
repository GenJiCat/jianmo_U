#!/usr/bin/env python3
"""
mlp_5layer.py

构建一个 MLP（5 个线性层），默认用 ReLU 作为中间激活函数。
默认示例：input_dim=4, layer_sizes=[16, 12, 8, 6, 3] 表示：
  Linear(4->16) -> ReLU -> Linear(16->12) -> ReLU -> Linear(12->8) -> ReLU
  -> Linear(8->6) -> ReLU -> Linear(6->3) (输出层)
"""

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP5(nn.Module):
    def __init__(self, input_dim: int, layer_sizes: List[int]):
        super().__init__()
        if len(layer_sizes) != 5:
            raise ValueError("layer_sizes must be a list of length 5 (five linear layers).")
        layers = []
        in_dim = input_dim
        for i, out_dim in enumerate(layer_sizes):
            layers.append(nn.Linear(in_dim, out_dim))
            # 对于最后一层不要加入激活（通常由 loss/上层处理 softmax 等）
            if i < 4:
                layers.append(nn.ReLU())
            in_dim = out_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# --- 默认示例模型（方便导入与自动识别脚本） ---
# 你可以修改 input_dim / layer_sizes 来适配你的数据
input_dim = 4
layer_sizes = [16, 12, 8, 6, 3]   # 五层输出尺寸（总共五个 Linear 层）
model = MLP5(input_dim=input_dim, layer_sizes=layer_sizes)

# --- 小示例：在 Iris 数据集上做一次短训练（可选） ---
def quick_train_demo(model, epochs=50, lr=0.01):
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    import torch.optim as optim

    iris = datasets.load_iris()
    X = iris.data.astype(float)
    y = iris.target.astype(int)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(X_train)
        loss = criterion(out, y_train)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}  loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        preds = model(X_test).argmax(dim=1)
        acc = (preds == y_test).float().mean().item()
        print(f"Test accuracy: {acc:.4f}")

if __name__ == "__main__":
    print("Model summary:")
    print(model)
    # 运行简短训练示例（如果你想测试）
    quick_train_demo(model, epochs=80, lr=0.01)
