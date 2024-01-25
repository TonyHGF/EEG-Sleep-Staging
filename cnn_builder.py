import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import skorch
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import svm


class CNN_Sleeping(nn.Module):
  def __init__(self, channels):
    super(CNN_Sleeping, self).__init__()
    self.conv1 = nn.Sequential(
        nn.Conv1d(
            in_channels = 1,
            out_channels = channels,
            kernel_size = 3,
            stride = 1,
            padding = 1,
        ),
        nn.ReLU()
    )
    self.conv2 = nn.Sequential(
        nn.Conv1d(
            in_channels = channels,
            out_channels = channels,
            kernel_size = 3,
            stride = 1,
            padding = 1,
        ),
        nn.ReLU()
    )
    self.conv3 = nn.Sequential(
        nn.Conv1d(
            in_channels = channels,
            out_channels = channels,
            kernel_size = 3,
            stride = 1,
            padding = 1,
        ),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2, stride=2)
    )
    self.l = nn.Sequential(
      nn.Linear(
        in_features = channels * 41,
        out_features = 7
      )
    )
  
  def forward(self, data):
    data = self.conv1(data)
    data = self.conv2(data)
    data = self.conv3(data)
    data = data.view(data.size()[0], -1)
    data = self.l(data)
    return data


def train_cnn_model(X, y):
    learning_rate = 2e-2
    channel = 32
    print(f"Start with learning rate: {learning_rate} and channel: {channel}")
    train_data,test_data,train_label,test_label = train_test_split(X,y,random_state=1,train_size=0.7,test_size=0.3)
    train_data = torch.Tensor(train_data)
    train_data = torch.unsqueeze(train_data, 1)
    test_data = torch.Tensor(test_data)
    test_data = torch.unsqueeze(test_data, 1)
    
    cnn = CNN_Sleeping(channels = channel)
    model = skorch.NeuralNetClassifier(cnn, criterion=torch.nn.CrossEntropyLoss,
                             device="cuda",
                             optimizer=torch.optim.SGD ,
                             lr=learning_rate,
                             max_epochs=120,
                             batch_size=128,
                             callbacks=[skorch.callbacks.EarlyStopping(lower_is_better=True)])
    
    model.fit(train_data, np.asarray(train_label, dtype=np.int64))
    train_score = model.score(train_data, np.asarray(train_label, dtype=np.int64))
    print("training set score:",train_score)
    
    # find each accuracy of each class
    pred = model.predict(test_data)
    pred = pred.tolist()
    test_label = test_label.tolist()
    cnt = len(test_label)
    acc = np.zeros(7)
    for i in range(cnt):
        if test_label[i] == pred[i]:
            acc[int(test_label[i])] += 1
    for i in range(7):
        print(f"Accuracy of class {i}: {acc[i]/test_label.count(i)}")
    
    test_score = model.score(test_data, np.asarray(test_label, dtype=np.int64))
    print("testing set score:",test_score)