import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# 1. CSV 로드
df = pd.read_csv(r"D:\대학\25 고대 3-1\위티\위티딥러닝세션\log.csv")


X = df[['A', 'B', 'C', 'D', 'E']].values  # 입력 피처 5개
y = df['score'].values                   # 레이블 (회귀 대상)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1)  # (batch, 1) 형태로 reshape

# 2. Dataset 정의
class LogDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = LogDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 3. MLP 모델 (회귀용)
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  # 🔥 회귀 → 출력 1개

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # 🔥 회귀니까 softmax 없음

model = MLP(input_dim=5)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()  # 🔥 회귀용 loss

# 4. 학습 루프
epoch_list = []
loss_list = []

for epoch in range(1, 21):
    model.train()
    total_loss = 0

    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        output = model(batch_X)
        loss = loss_fn(output, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    epoch_list.append(epoch)
    loss_list.append(avg_loss)
    print(f"[Epoch {epoch}] Loss: {avg_loss:.4f}")

# 5. 시각화
plt.plot(epoch_list, loss_list, marker='o')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss over Epochs')
plt.grid()
plt.show()
