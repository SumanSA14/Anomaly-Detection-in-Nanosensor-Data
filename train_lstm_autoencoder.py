import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


SEQ_LEN = 10
BATCH_SIZE = 32
EPOCHS = 50
HIDDEN_SIZE = 64
LR = 1e-3


os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("plots", exist_ok=True)


df = pd.read_csv("dataset_final.csv")
df = df[['Temperature', 'Humidity', 'Air Quality', 'Light', 'Loudness']]
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df.values)

def create_sequences(data, seq_len):
    xs = []
    for i in range(len(data) - seq_len):
        x = data[i:(i + seq_len)]
        xs.append(x)
    return torch.tensor(np.array(xs), dtype=torch.float32)

X = create_sequences(scaled, SEQ_LEN)
dataset = DataLoader(X, batch_size=BATCH_SIZE, shuffle=True)

class LSTMAE(nn.Module):
    def __init__(self, n_features, hidden_size):
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, n_features, batch_first=True)

    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        decoder_input = hidden.repeat(x.size(1), 1, 1).permute(1, 0, 2)
        output, _ = self.decoder(decoder_input)
        return output


n_features = X.shape[2]
model = LSTMAE(n_features=n_features, hidden_size=HIDDEN_SIZE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


log_file = os.path.join("logs", "training_metrics.csv")
with open(log_file, "w") as f:
    f.write("epoch,average_loss\n")


epoch_losses = []
for epoch in range(EPOCHS):
    model.train()
    losses = []

    for batch in dataset:
        pred = model(batch)
        loss = criterion(pred, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    avg_loss = sum(losses) / len(losses)
    epoch_losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.6f}")

 
    with open(log_file, "a") as f:
        f.write(f"{epoch+1},{avg_loss:.6f}\n")


torch.save(model.state_dict(), "models/lstm_autoencoder.pt")

import joblib
joblib.dump(scaler, "models/scaler.pkl")

plt.figure(figsize=(8, 5))
plt.plot(range(1, EPOCHS+1), epoch_losses, marker='o')
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/loss_curve.png")
plt.close()
