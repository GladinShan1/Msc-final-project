# scripts/evaluate_model.py

import os
import sys
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score

# Add parent directory to the path so 'scripts' module can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import utility functions
from scripts.utils import preprocess_features, encode_target

def evaluate():
    # Load unseen data
    df = pd.read_csv('data/Last_50_Forex_Data_Entries.csv', parse_dates=['Date'])

    # Prepare features and true labels
    X_unseen = preprocess_features(df)
    y_true = encode_target(df['finbert_sentiment'])

    # Load trained model
    model = joblib.load('models/forex_model.pkl')

    # Predict
    y_pred = model.predict(X_unseen)

    # Evaluate
    print("🔍 Classification Report:")
    report = classification_report(y_true, y_pred)
    print(report)
    print("✅ Accuracy:", accuracy_score(y_true, y_pred))

    # Save report
    os.makedirs('outputs/reports', exist_ok=True)
    with open('outputs/reports/classification_report_unseen.txt', 'w') as f:
        f.write(report)

    # Add predictions to DataFrame
    df['Model_Prediction'] = y_pred
    df['Signal'] = df['Model_Prediction'].map({1: 'BUY', 0: 'HOLD', -1: 'SELL'})

    # Save prediction CSV
    os.makedirs('outputs/signals', exist_ok=True)
    df[['Date', 'Signal']].to_csv('outputs/signals/unseen_signals.csv', index=False)
    print("✅ Signal predictions saved to outputs/signals/unseen_signals.csv")

    # Plot last 30 signals
    last_df = df[['Date', 'Signal']].tail(30).copy()
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=last_df, x='Date', y='Signal', hue='Signal', palette='Set2', s=100)
    plt.title('Model Signals on Last 30 Entries')
    plt.xticks(rotation=45)
    plt.tight_layout()

    os.makedirs('outputs/plots', exist_ok=True)
    plt.savefig('outputs/plots/signal_chart.png')
    print("📊 Signal plot saved to outputs/plots/signal_chart.png")

if __name__ == "__main__":
    evaluate()
