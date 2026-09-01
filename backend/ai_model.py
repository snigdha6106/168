import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os

class VelocityEstimator(nn.Module):
    def __init__(self, input_size=6, hidden_size=32, num_layers=2):
        super(VelocityEstimator, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        out, _ = self.lstm(x)
        # We only care about the last time step prediction in streaming mode,
        # but for training we might predict all steps.
        out = self.fc(out) 
        return out

def train_model(data_path="data/train_route_processed.csv", model_path="model_weights.pth"):
    print("Training AI Speed & Vibration Filter...")
    df = pd.read_csv(data_path)
    df = df.dropna(subset=['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'true_velocity'])
    
    X = df[['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']].values
    y = df['true_velocity'].values
    
    # Normalize Inputs and Targets
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X = (X - X_mean) / (X_std + 1e-8)
    
    y_mean = y.mean()
    y_std = y.std()
    y_norm = (y - y_mean) / (y_std + 1e-8)
    
    seq_len = 10
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y_norm[i+seq_len-1])
        
    X_seq = torch.tensor(X_seq, dtype=torch.float32)
    y_seq = torch.tensor(y_seq, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
    
    model = VelocityEstimator()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    model.train()
    for epoch in range(500): # Deep training to learn actual braking patterns from IMU
        optimizer.zero_grad()
        outputs = model(X_seq)
        loss = criterion(outputs[:, -1, :], y_seq[:, 0, :])
        loss.backward()
        optimizer.step()
        
    print(f"Finished training. Final Loss: {loss.item():.4f}")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'X_mean': X_mean,
        'X_std': X_std,
        'y_mean': y_mean,
        'y_std': y_std
    }, model_path)
    print(f"Model saved to {model_path}")
    return model, X_mean, X_std, y_mean, y_std

class AIFilterInference:
    def __init__(self, model_path="model_weights.pth", seq_len=10):
        self.seq_len = seq_len
        self.model = VelocityEstimator()
        self.buffer = []
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.X_mean = checkpoint['X_mean']
            self.X_std = checkpoint['X_std']
            self.y_mean = checkpoint.get('y_mean', 15.0)
            self.y_std = checkpoint.get('y_std', 1.0)
            print("Loaded pre-trained AI model.")
        else:
            print("No pre-trained model found! Running training...")
            if not os.path.exists("data/train_route_processed.csv"):
                print("Real dataset not found! Please run parse_iovnbd.py first.")
                return
            _, self.X_mean, self.X_std, self.y_mean, self.y_std = train_model(model_path=model_path)
            
        self.model.eval()

    def predict(self, imu_data):
        # Normalize
        norm_data = (np.array(imu_data) - self.X_mean) / (self.X_std + 1e-8)
        self.buffer.append(norm_data)
        
        if len(self.buffer) > self.seq_len:
            self.buffer.pop(0)
            
        if len(self.buffer) < self.seq_len:
            return self.y_mean # Fallback to mean if not enough data
            
        with torch.no_grad():
            x_tensor = torch.tensor([self.buffer], dtype=torch.float32)
            out = self.model(x_tensor)
            pred_norm = out[0, -1, 0].item()
            return pred_norm * self.y_std + self.y_mean

if __name__ == "__main__":
    train_model()
