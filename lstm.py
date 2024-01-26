import numpy as np
import torch
from torch import nn
import os
from tqdm.notebook import tqdm

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, self.hidden_size).to(x.device) 
        c0 = torch.zeros(self.num_layers * 2, self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out)
        return out

# Prepare data
inputs = []
outputs = []
for dirname, _, filenames in os.walk('./result'):
    for filename in filenames:
        file_path = os.path.join(dirname, filename)
        file = np.load(file_path)
        inputs.append(file['features'])
        outputs.append(file['labels'])
N = len(inputs)

# Parameters
hidden_size = 256
input_size = 105
output_size = 7
num_layers = 2  # You can change this
num_epochs = 500  # You can change this
learning_rate = 0.01  # You can change this

# Instantiate the model
model = LSTMModel(input_size, hidden_size, num_layers, output_size)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
model.to(device='cuda')
criterion.to(device='cuda')

# Training loop
for epoch in tqdm(range(num_epochs)):
    total_loss=0
    for i in range(N):
        input = torch.tensor(inputs[i], dtype=torch.float32).to(device='cuda')
        output = torch.tensor(outputs[i], dtype=torch.long).to(device='cuda')
        output_pred = model(input)
        optimizer.zero_grad()
        loss = criterion(output_pred, output)
        total_loss += loss
        loss.backward()
        optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss.item()}')
