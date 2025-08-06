import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, classification_report as report_metrics

# --- Ensure output folders exist
os.makedirs("outputs/reports", exist_ok=True)
os.makedirs("outputs/plots", exist_ok=True)
os.makedirs("models", exist_ok=True)

# --- Load dataset
df = pd.read_csv("data/Merged_Forex_Dataset.csv", parse_dates=['Date'])

# --- Prepare features
X = df.drop(columns=[
    'Date', 'ticker_x', 'true_sentiment', 'title', 'author', 'url',
    'source', 'text', 'finbert_sentiment', 'ticker_y', 'country'
], errors='ignore')
X = X.select_dtypes(include=[np.number]).fillna(0)

# --- Prepare target
y = df['finbert_sentiment'].map({'Positive': 1, 'Neutral': 0, 'Negative': -1})

# --- Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# --- Train Logistic Regression
model = LogisticRegression(max_iter=1000, multi_class='ovr')
model.fit(X_train, y_train)

# --- Evaluate
y_pred = model.predict(X_test)
report_text = classification_report(y_test, y_pred)
print("✅ Logistic Regression Report:\n", report_text)

# --- Save classification report
report_dict = report_metrics(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
report_df.to_csv("outputs/reports/classification_report_logistic.csv")
print("📁 Report saved to outputs/reports/classification_report_logistic.csv")

# --- Save signal predictions
if 'Date' in df.columns:
    pd.DataFrame({
        'Date': df.iloc[y_test.index]['Date'].values,
        'Prediction': y_pred
    }).to_csv('outputs/reports/train_signals_logistic.csv', index=False)
    print("📁 Signal predictions saved to outputs/reports/train_signals_logistic.csv")

# --- Save model
joblib.dump(model, "models/logistic_model.pkl")
print("✅ Logistic Regression model saved to models/logistic_model.pkl")

# --- Feature Importance
coefficients = model.coef_
feature_names = X.columns

# For multi-class, average absolute coefficients across classes
if coefficients.shape[0] > 1:
    avg_importance = np.mean(np.abs(coefficients), axis=0)
else:
    avg_importance = np.abs(coefficients[0])

feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': avg_importance
}).sort_values(by='Importance', ascending=False)

# --- Save feature importance plot
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance.head(20), x='Importance', y='Feature', palette='viridis')
plt.title('Top Feature Importances (Logistic Regression)')
plt.tight_layout()
plt.savefig("outputs/plots/feature_importance_logistic.png")
plt.close()
print("📊 Feature importance plot saved to outputs/plots/feature_importance_logistic.png")
