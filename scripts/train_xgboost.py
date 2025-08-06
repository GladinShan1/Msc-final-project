import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, classification_report as report_metrics

# --- Step 1: Ensure output folders exist
os.makedirs("outputs/reports", exist_ok=True)
os.makedirs("outputs/plots", exist_ok=True)
os.makedirs("models", exist_ok=True)

# --- Step 2: Load dataset
df = pd.read_csv("data/Merged_Forex_Dataset.csv", parse_dates=['Date'])

# --- Step 3: Prepare features
X = df.drop(columns=[
    'Date', 'ticker_x', 'true_sentiment', 'title', 'author', 'url',
    'source', 'text', 'finbert_sentiment', 'ticker_y', 'country'
], errors='ignore')

X = X.select_dtypes(include=[np.number]).fillna(0)

# --- Step 4: Prepare target for XGBoost (labels must start at 0)
y = df['finbert_sentiment'].map({'Negative': 0, 'Neutral': 1, 'Positive': 2})

# --- Step 5: Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# --- Step 6: Train XGBoost model
model = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
model.fit(X_train, y_train)

# --- Step 7: Evaluate model
y_pred = model.predict(X_test)
report_text = classification_report(y_test, y_pred)
print("✅ Classification Report:\n", report_text)

# --- Save classification report (CSV)
report_dict = report_metrics(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
report_df.to_csv("outputs/reports/classification_report_xgboost.csv")
print("📁 Classification report saved to outputs/reports/classification_report_xgboost.csv")

# --- Save signal predictions
if 'Date' in df.columns:
    reverse_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
    predicted_labels = pd.Series(y_pred).map(reverse_map)
    pd.DataFrame({
        'Date': df.iloc[y_test.index]['Date'].values,
        'Prediction': predicted_labels
    }).to_csv('outputs/reports/train_signals_xgboost.csv', index=False)
    print("📁 XGBoost signal predictions saved to outputs/reports/train_signals_xgboost.csv")

# --- Step 8: Save trained model
joblib.dump(model, "models/xgboost_model.pkl")
print("✅ XGBoost model saved to: models/xgboost_model.pkl")

# --- Step 9: Feature Importance Plot
importances = model.feature_importances_
feature_names = X.columns

fi_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=fi_df.head(15), palette='plasma')
plt.title("Top 15 Feature Importances (XGBoost)")
plt.tight_layout()
plt.savefig("outputs/plots/feature_importance_xgboost.png")
print("📊 Feature importance chart saved to outputs/plots/feature_importance_xgboost.png")
