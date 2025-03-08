import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import argparse
from model import TimeSeriesTransformer

def parse_args():
    parser = argparse.ArgumentParser(description='Time Series Transformer Training')
    parser.add_argument('--data_path', type=str, default="",
                        help='Path to the data file')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Input batch size for training')
    parser.add_argument('--window_length', type=int, default=5,
                        help='Window length for sequence creation')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--test_size', type=float, default=0.4,
                        help='')
    parser.add_argument('--val_size', type=float, default=0.5,
                        help='')
    parser.add_argument('--input_size', type=int, default=2,
                        help='Number of features in input data')
    parser.add_argument('--no_plot', action='store_true',
                        help='Disable plotting')
    return parser.parse_args()

def create_sequences(data, window_length):
    xs, ys = [], []
    for i in range(len(data)-window_length):
        x = data[i:(i+window_length), :2]
        y = data[i+window_length, 1]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def prepare_data(data_path, window_length, test_size, val_size):
    df = pd.read_excel(data_path)
    data = df.values
    X, y = create_sequences(data, window_length)
    scaler = StandardScaler()
    X = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size, shuffle=False)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs):
    train_losses, val_losses = [], []
    for epoch in range(epochs):
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

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                output = model(X_val)
                val_loss += criterion(output.squeeze(), y_val).item()
        val_losses.append(val_loss/len(val_loader))

        print(f'Epoch {epoch+1:03}: Train Loss {train_losses[-1]:.4f}, Val Loss {val_losses[-1]:.4f}')
    
    return train_losses, val_losses

def evaluate_model(model, test_loader):
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
    
    return test_preds, test_true, mse.item(), mae.item(), r2.item()

def main():
    args = parse_args()
    
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(
        args.data_path, args.window_length, args.test_size, args.val_size
    )
    
    train_data = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_data = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    test_data = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)
    test_loader = DataLoader(test_data, batch_size=args.batch_size)
    
    model = TimeSeriesTransformer(
        input_size=args.input_size,
        seq_length=args.window_length
    )
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, args.epochs
    )
    
    test_preds, test_true, mse, mae, r2 = evaluate_model(model, test_loader)
    
    print(f'\nTest Results:')
    print(f'MSE: {mse:.4f}')
    print(f'MAE: {mae:.4f}')
    print(f'R² Score: {r2:.4f}')
    
    if not args.no_plot:
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.title('Training Curves')
        
        plt.subplot(1, 2, 2)
        plt.plot(test_true, label='True Values')
        plt.plot(test_preds, label='Predictions')
        plt.xlabel('Time Step')
        plt.ylabel('preN103 Value')
        plt.legend()
        plt.title('Test Set Predictions')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()