import pandas as pd
import matplotlib.pyplot as plt

def detect_interval(df):
    diffs = df.index.to_series().diff().dropna()
    if diffs.empty:
        return "unknown"

    most_common = diffs.mode()[0]

    if most_common <= pd.Timedelta(minutes=1, seconds=30):
        return "minute"
    elif most_common <= pd.Timedelta(hours=1, minutes=30):
        return "hour"
    else:
        return "day"


def plot_data(stock_data, stock, single_col = False, title = None):
    interval = stock_data.index.to_numpy()
    
    if not single_col:
        features = stock_data.columns.to_list()
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
        count = 0
        for row in range(2):
            for col in range(2):
                axes[row][col].plot(interval, stock_data[features[count]])
                axes[row][col].set_title(features[count])
                count +=1
        fig.suptitle(f'{stock} prices by interval: {detect_interval(stock_data)}')
    else:
        plt.plot(interval, stock_data)
        plt.title(f'{stock} {title} by interval: {detect_interval(stock_data)}')