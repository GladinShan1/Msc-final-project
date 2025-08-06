# Placeholder for utility functions
# scripts/utils.py

import pandas as pd
import numpy as np

def preprocess_features(df):
    drop_cols = [
        'Date', 'ticker_x', 'true_sentiment', 'title', 'author', 'url',
        'source', 'text', 'finbert_sentiment', 'ticker_y', 'country'
    ]
    X = df.drop(columns=drop_cols, errors='ignore')
    X = X.select_dtypes(include=[np.number])
    X = X.fillna(0)
    return X

def encode_target(sentiment_series):
    return sentiment_series.map({'Positive': 1, 'Neutral': 0, 'Negative': -1})


