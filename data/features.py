import numpy as np
import pandas as pd


def compute_returns(prices):
    returns = np.zeros(len(prices))
    returns[0] = np.nan
    for i in range(1,len(prices)):
        returns[i] = (prices[i]-prices[i-1])/prices[i-1]
    return returns

def compute_moving_average(data, window):
        ma = np.zeros(len(data))
        ma[:window] = np.nan
        for i in range(window, len(data)):
            data_window = data[i-window:i]
            ma[i] = np.mean(data_window)
        return ma


def prices_to_returns(prices):
        returns = np.zeros(len(prices)-1)
        for i in range(1,len(prices)):
            returns[i-1] = (prices[i]-prices[i-1])/prices[i-1]
        return returns


def compute_day_of_week_feature(data: pd):
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex")
        return data.index.dayofweek.to_numpy()

def compute_rolling_std(data, window):
        rolling_std = np.zeros(len(data))
        rolling_std[:window] = np.nan
        for i in range(window, len(data)):
            data_window = data[i-window:i]
            rolling_std[i] = np.std(data_window)
        return rolling_std

def compute_target(data):
    target = np.zeros(len(data))
    for i in range(len(data)-1):
        target[i] = 1 if data[i+1]>data[i] else 0
    return target

