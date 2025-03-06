import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from model import TimeSeriesTransformer


batch_size = 32
windows_length = 5

df = pd.read_excel("C:\\Users\\86136\\Desktop\\LSTM\\134.xlsx")
data = df.values

def create_sequences(data, windows_length):
    xs, ys = [], []
    for i in range(len(data)-windows_length):
        x = data[i:(i+windows_length), :]
        y = data[i+windows_length, 2] 
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


X, y = create_sequences(data, windows_length)

# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

# 划分数据集
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, shuffle=False)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.6, shuffle=False)

# 转换为Tensor
train_data = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
val_data = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
test_data = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))


train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)


model = TimeSeriesTransformer(input_size=3, seq_length=windows_length)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 训练过程
train_losses, val_losses = [], []
for epoch in range(100):
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output.squeeze(), y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    train_losses.append(epoch_loss/len(train_loader))

    # 验证
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_val, y_val in val_loader:
            output = model(X_val)
            val_loss += criterion(output.squeeze(), y_val).item()
    val_losses.append(val_loss/len(val_loader))

    print(f'Epoch {epoch+1}: Train Loss {train_losses[-1]:.4f}, Val Loss {val_losses[-1]:.4f}')

# 测试
model.eval()
test_preds, test_true = [], []
with torch.no_grad():
    for X_test, y_test in test_loader:
        output = model(X_test)
        output_clipped = torch.clamp(output, min=0.0)
        test_preds.extend(output_clipped.squeeze().tolist())
        test_true.extend(y_test.tolist())
test_preds = torch.FloatTensor(test_preds)
test_true = torch.FloatTensor(test_true)
mse = nn.MSELoss()(test_preds, test_true)
mae = nn.L1Loss()(test_preds, test_true)
ss_tot = torch.sum((test_true - torch.mean(test_true))**2)
ss_res = torch.sum((test_true - test_preds)**2)
r2 = 1 - ss_res / ss_tot

print(f'\nTest Results:')
print(f'MSE: {mse.item():.4f}')
print(f'MAE: {mae.item():.4f}')
print(f'R² Score: {r2.item():.4f}')
# 可视化训练过程
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

# 可视化预测结果
plt.figure(figsize=(12, 6))
plt.plot(test_true, label='True Values')
plt.plot(test_preds, label='Predictions')
plt.xlabel('Time Step')
plt.ylabel('preN103 Value')
plt.legend()
plt.title('Test Set Predictions vs True Values')
plt.show()