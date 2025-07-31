from torch import nn
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

def train_step(model:nn.Module, optimizer, loss_fn, data_loader: DataLoader):
    train_loss = 0
    model.train()
    acc = 0
    for batch_X, batch_y in data_loader:
        batch_y = batch_y.unsqueeze(1)
        
        
        logits = model(batch_X)
        acc += binary_accuracy(logits, batch_y)
        # print("Logits mean:", logits.mean().item(), "std:", logits.std().item())
        # break


        loss = loss_fn(logits, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss
    
    avg_train_loss = train_loss/len(data_loader)
    avg_acc = acc/len(data_loader)
    return avg_train_loss,avg_acc



def test_step(model:nn.Module, loss_fn, data_loader: DataLoader):
    test_loss = 0
    model.eval()
    acc = 0
    with torch.inference_mode():
        for batch_X, batch_y in data_loader:
            batch_y = batch_y.unsqueeze(1)
            logits = model(batch_X)
            acc += binary_accuracy(logits, batch_y)
            test_loss += loss_fn(logits, batch_y)
        average_test_loss = test_loss/len(data_loader)
        avg_acc = acc/len(data_loader)

    return average_test_loss, avg_acc


def binary_accuracy(y_pred_logits, y_true):
    y_prob = torch.sigmoid(y_pred_logits)
        
    # Threshold at 0.5 to get predicted classes
    y_pred = (y_prob >= 0.50).float()
    
    # Compare to true labels
    correct = (y_pred == y_true).sum().item()
    total = y_true.size(0)
    
    accuracy = correct / total
    return accuracy



def split_train_test(data_set: pd.DataFrame, split_ratio):

    split_point = int(len(data_set) * split_ratio)

    X_train = data_set.iloc[:split_point, :]
    X_test = data_set.iloc[split_point:, :]

    return X_train, X_test

def create_series(features, time_step,target_col='Returns'):
    direction = (features[target_col]>0).astype(int)
    X = []
    y = []
    
    for i in range(len(features)-time_step):
        X.append(features.iloc[i:(i+time_step),:])
        y.append(direction[i+time_step])
    return np.array(X), np.array(y)

    