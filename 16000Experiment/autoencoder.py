# autoencoder.py
# autoencoder.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(133, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 133),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def train_autoencoder(input_file, n_epochs=15, batch_size=32, learning_rate=0.001):
    df = pd.read_csv(input_file, header=None)
    numerical_data = df.values
    scaler = MinMaxScaler()
    numerical_data_normalized = scaler.fit_transform(numerical_data)
    
    X_train, X_val = train_test_split(numerical_data_normalized, test_size=0.2, random_state=42)
    X_train_tensor = torch.tensor(X_train.astype(np.float32))
    X_val_tensor = torch.tensor(X_val.astype(np.float32))
    train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, X_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = Autoencoder()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        for data in train_loader:
            inputs, _ = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        print(f'Epoch {epoch+1}/{n_epochs}, Loss: {train_loss/len(train_loader.dataset)}')

    encoded_data = model.encoder(torch.tensor(numerical_data_normalized.astype(np.float32))).detach().numpy()
    encoded_df = pd.DataFrame(encoded_data)
    encoded_df.to_csv("encoded_data.csv", index=False)
    print("Autoencoder training completed and data encoded successfully.")
    return encoded_df

if __name__ == "__main__":
    input_file = "merged_configuration_all_processed.csv"
    train_autoencoder(input_file)
