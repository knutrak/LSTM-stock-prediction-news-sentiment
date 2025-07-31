# LSTM Stock Prediction with News Sentiment

This project explores the use of LSTM (Long Short-Term Memory) neural networks to predict stock prices based on historical stock data and sentiment analysis of financial news headlines. The aim is to improve predictive performance by incorporating external sentiment signals in addition to traditional technical indicators.

## Data Sources

- **Stock data:** Historical data containing Open, High, Low, Close, Volume (OHLCV).
- **Sentiment data:** News headlines related to Apple (AAPL), scraped and analyzed using trained finbert model('ProsusAI/finbert) sentiment scoring.

## Features Used

- Technical indicators:
  - Returns
  - Moving Averages (MA 10, MA 50, MA 100)
  - Rolling standard deviation (20-day)
  - RSI (Relative Strength Index)
  - EMA (Exponential Moving Average)
  - Day of the week
- Sentiment-based indicators:
  - `avg_sentiment`: average sentiment score per day
  - `has_news`: binary flag for whether a news article was present
  - `sentiment_moving_avg`, `sentiment_diff`, `sentiment_sign`: engineered sentiment trend features

## Model Architecture

- LSTM (Long Short-Term Memory) neural network built using PyTorch.
- Sequence-to-one prediction to classify the direction of price movement.
- Separate training and validation splits with logging of accuracy and loss.

## Training Status

- The model currently **overfits** to the training set and does not generalize well to the test set.
- Sentiment features are added and show some correlation but need further tuning.
- Still **a work in progress** — further improvements needed in regularization, feature engineering, and model tuning.

