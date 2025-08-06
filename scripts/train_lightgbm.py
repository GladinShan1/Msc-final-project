# scripts/train_lightgbm.py

import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
DATA_PATH = "data/Merged_Forex_Dataset.csv"
MODEL_PATH = "models/lightgbm_model.pkl"
SIGNAL_OUTPUT = "outputs/reports/train_signals_lightgbm.csv"
REPORT_OUTPUT = "outputs/reports/classification_report_lightgbm.csv"
FI_PLOT_PATH = "outputs/plots/feature_importance_lightgbm.png"

# Load dataset
df = pd.read_csv(DATA_PATH, parse_dates=["Date"])

# Prepare features
X = df.drop(columns=[
    'Date', 'ticker_x', 'true_sentiment', 'title', 'author', 'url',
    'source', 'text', 'finbert_sentiment', 'ticker_y', 'country'
], errors='ignore')
X = X.select_dtypes(include=[np.number]).fillna(0)

# Prepare target
y = df['finbert_sentiment'].map({'Positive': 2, 'Neutral': 1, 'Negative': 0})

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Predict & evaluate
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv(REPORT_OUTPUT)
print("✅ Classification Report:")
print(report_df)

# Save model
joblib.dump(model, MODEL_PATH)
print(f"📁 LightGBM model saved to: {MODEL_PATH}")

# Save full signal predictions
predictions = model.predict(X)
signal_df = pd.DataFrame({
    'Date': df['Date'],
    'Prediction': predictions
})
signal_df['Prediction'] = signal_df['Prediction'].map({2: 1, 1: 0, 0: -1})  # back to original signal format
signal_df.to_csv(SIGNAL_OUTPUT, index=False)
print(f"📁 Signal predictions saved to {SIGNAL_OUTPUT}")

# Feature importance
fi = model.feature_importances_
fi_df = pd.DataFrame({'Feature': X.columns, 'Importance': fi}).sort_values(by='Importance', ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=fi_df.head(15), palette='viridis')
plt.title("Top Feature Importances (LightGBM)")
plt.tight_layout()
plt.savefig(FI_PLOT_PATH)
print(f"📊 Feature importance chart saved to {FI_PLOT_PATH}")
