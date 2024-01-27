import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm.notebook import tqdm

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, outputs, targets):
        # 自定义损失计算
        return nn.functional.binary_cross_entropy_with_logits(outputs, targets, pos_weight=self.pos_weight)

# Function to train logistic regression model in PyTorch
def train_logistic_model_pytorch(X, y, pos_weight, learning_rate=0.01, epochs=2000):
    # Convert numpy arrays to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    # Create the logistic regression model
    input_dim = X_tensor.shape[1]
    model = LogisticRegressionModel(input_dim)
    criterion = CustomLoss()  # 使用自定义损失函数
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in tqdm(range(epochs)):
        # Forward pass
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model

'''
Example usage (assuming X and y are defined):
model = train_logistic_model_pytorch(X, y)

This code will train a logistic regression model using PyTorch.
Note: For real data, you should split the data into training and testing sets and evaluate the model accordingly.
'''

def predict(model, X):
    # 将输入数据 X 转换为 PyTorch 张量
    X_tensor = torch.tensor(X, dtype=torch.float32)
    
    # 确保模型处于评估模式
    model.eval()

    # 通过模型获得预测结果
    with torch.no_grad():  # 确保不计算梯度
        outputs = model(X_tensor)
    
    # 将输出转换为二进制标签（0或1）
    predicted_labels = (outputs > 0.5).int()
    
    return predicted_labels