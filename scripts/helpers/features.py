# helpers/features.py

def preprocess_features(df):
    # Drop non-numeric and irrelevant columns
    features = df.copy()
    features = features.drop(columns=['Date', 'finbert_sentiment'], errors='ignore')
    numeric_features = features.select_dtypes(include=['int64', 'float64'])
    return numeric_features

def encode_target(sentiment_series):
    mapping = {'BUY': 1, 'HOLD': 0, 'SELL': -1}
    return sentiment_series.map(mapping)
