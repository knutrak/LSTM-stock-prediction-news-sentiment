import numpy as np
import pandas as pd
import torch
from datetime import datetime
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from torch import nn
import torch.optim as optim

class LSTM_Model_Regression(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout = 0.4):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        
        self.nn_nodes = 16
        self.nnLayers = nn.Sequential(
            nn.Linear(hidden_size, out_features=self.nn_nodes),
            nn.ReLU(),
            nn.Linear(in_features=self.nn_nodes, out_features=self.nn_nodes*2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(in_features=self.nn_nodes*2,out_features=output_size)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.nnLayers(out[:, -1, :])
        return out
    

class LSTM_Model_Classification(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout = 0.4):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        
        self.nn_nodes = 8
        self.nnLayers = nn.Sequential(
            nn.Linear(hidden_size, out_features=self.nn_nodes),
            nn.BatchNorm1d(self.nn_nodes),
            nn.ReLU(),
            nn.Linear(in_features=self.nn_nodes, out_features=self.nn_nodes*2),
            nn.BatchNorm1d(self.nn_nodes*2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=self.nn_nodes*2,out_features=output_size)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.nnLayers(out[:, -1, :])
        return out
    




