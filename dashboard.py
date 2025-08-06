# dashboard.py

import streamlit as st
import pandas as pd
import os
import plotly.express as px

# --- File paths
report_path_rf = 'outputs/reports/classification_report.csv'
feature_importance_lr = 'outputs/plots/feature_importance_logistic.png'

signal_path_rf = 'outputs/reports/train_signals.csv'
report_path_lr = 'outputs/reports/classification_report_logistic.csv'
signal_path_lr = 'outputs/reports/train_signals_logistic.csv'
signal_path_xgb = 'outputs/reports/train_signals_xgboost.csv'
report_path_xgb = 'outputs/reports/classification_report_xgboost.csv'
signal_path_lgb = 'outputs/reports/train_signals_lightgbm.csv'
report_path_lgb = 'outputs/reports/classification_report_lightgbm.csv'

model_path_rf = 'models/forex_model.pkl'
model_path_lr = 'models/logistic_model.pkl'
model_path_xgb = 'models/xgboost_model.pkl'
model_path_lgb = 'models/lightgbm_model.pkl'

feature_importance_path = 'outputs/plots/feature_importance.png'
feature_importance_xgb = 'outputs/plots/feature_importance_xgboost.png'
feature_importance_lgb = 'outputs/plots/feature_importance_lightgbm.png'

merged_dataset_path = 'data/Merged_Forex_Dataset.csv'
unseen_dataset_path = 'data/Last_50_Forex_Data_Entries.csv'
github_link = "https://colab.research.google.com/drive/1A6kGTS9MdvkthEQhevl_nXxUKU2S_AA7?usp=sharing"

# --- Streamlit UI
st.set_page_config(page_title="Forex Sentiment Dashboard", layout="wide")
st.title("📈 Forex Sentiment Signal Dashboard")
st.markdown("Built using FinBERT Sentiment + Macroeconomic Indicators")

# --- Tabs
tabs = st.tabs([
    "📊 RF Report", "📌 RF Signals", "📈 RF Signal Chart", "🧠 Feature Importance",
    "📊 Logistic Regression Report", "📌 LR Signals",
    "🤖 Model Comparison", "🧪 XGBoost Results", "🧪 LightGBM Results"
])

# --- Utility: Clean predictions
def clean_preds(df):
    df['Prediction'] = df['Prediction'].astype(str).str.strip()
    df['Prediction'] = df['Prediction'].replace({'BUY': 1, 'HOLD': 0, 'SELL': -1})
    df['Prediction'] = df['Prediction'].astype(int)
    return df

# --- Tab 1: RF Report
with tabs[0]:
    st.subheader("📊 Random Forest Classification Report")
    if os.path.exists(report_path_rf):
        df = pd.read_csv(report_path_rf, index_col=0)
        st.dataframe(df.style.format("{:.2f}"), use_container_width=True)
    else:
        st.warning("Report not found.")

# --- Tab 2: RF Signal Table
with tabs[1]:
    st.subheader("📌 RF Predicted Signals")
    if os.path.exists(signal_path_rf):
        df = pd.read_csv(signal_path_rf, parse_dates=['Date'])
        df['Prediction'] = df['Prediction'].replace({1: 'BUY', 0: 'HOLD', -1: 'SELL'})
        st.dataframe(df.tail(50).reset_index(drop=True), use_container_width=True)
    else:
        st.warning("Signal file not found.")

# --- Tab 3: RF Signal Chart
with tabs[2]:
    st.subheader("📈 RF Sentiment Chart")
    if os.path.exists(signal_path_rf):
        df = pd.read_csv(signal_path_rf, parse_dates=['Date'])
        df['Prediction'] = df['Prediction'].replace({1: 'BUY', 0: 'HOLD', -1: 'SELL'})
        fig = px.scatter(df, x='Date', y='Prediction', color='Prediction',
                         color_discrete_map={'BUY': 'green', 'HOLD': 'blue', 'SELL': 'red'},
                         title="RF Signals Over Time")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Signal data not found.")

# --- Tab 4: RF Feature Importance
with tabs[3]:
    st.subheader("🧠 Feature Importance (RF)")
    if os.path.exists(feature_importance_path):
        st.image(feature_importance_path,  width=800)
    else:
        st.warning("Feature importance not found.")

# --- Tab 5: Logistic Regression Report + Chart + Feature Importance
with tabs[4]:
    st.subheader("📊 Logistic Regression Report")
    if os.path.exists(report_path_lr):
        df = pd.read_csv(report_path_lr, index_col=0)
        st.dataframe(df.style.format("{:.2f}"), use_container_width=True)
    else:
        st.warning("Report not found.")

    if os.path.exists(signal_path_lr):
        df = pd.read_csv(signal_path_lr, parse_dates=['Date'])
        df['Prediction'] = df['Prediction'].replace({1: 'BUY', 0: 'HOLD', -1: 'SELL'})
        fig = px.scatter(df, x='Date', y='Prediction', color='Prediction',
                         color_discrete_map={'BUY': 'green', 'HOLD': 'blue', 'SELL': 'red'},
                         title="LR Signals Over Time")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Signal data not found.")

    # Logistic Regression Feature Importance
    st.subheader("🧠 Feature Importance (Logistic Regression)")
    if os.path.exists(feature_importance_lr):
        st.image(feature_importance_lr,  width=800)
    else:
        st.warning("Feature importance plot not found.")


# --- Tab 6: Logistic Regression Signals
with tabs[5]:
    st.subheader("📌 Logistic Regression Signals")
    if os.path.exists(signal_path_lr):
        df = pd.read_csv(signal_path_lr, parse_dates=['Date'])
        df['Prediction'] = df['Prediction'].replace({1: 'BUY', 0: 'HOLD', -1: 'SELL'})
        st.dataframe(df.tail(50).reset_index(drop=True), use_container_width=True)
    else:
        st.warning("Signal file not found.")

# --- Tab 7: Model Comparison
with tabs[6]:
    st.subheader("🤖 Model Prediction Comparison")
    try:
        rf = clean_preds(pd.read_csv(signal_path_rf))
        lr = clean_preds(pd.read_csv(signal_path_lr))
        xgb = clean_preds(pd.read_csv(signal_path_xgb))
        lgb = clean_preds(pd.read_csv(signal_path_lgb))

        # Align lengths
        min_len = min(len(rf), len(lr), len(xgb), len(lgb))
        rf, lr, xgb, lgb = rf[:min_len], lr[:min_len], xgb[:min_len], lgb[:min_len]

        # Compare
        rf_lr = (rf['Prediction'] == lr['Prediction'])
        rf_xgb = (rf['Prediction'] == xgb['Prediction'])
        rf_lgb = (rf['Prediction'] == lgb['Prediction'])
        lr_xgb = (lr['Prediction'] == xgb['Prediction'])
        lr_lgb = (lr['Prediction'] == lgb['Prediction'])
        xgb_lgb = (xgb['Prediction'] == lgb['Prediction'])
        all_match = rf_lr & rf_xgb & rf_lgb

        st.markdown(f"✅ **RF vs LR Match**: `{rf_lr.mean()*100:.2f}%`")
        st.markdown(f"✅ **RF vs XGB Match**: `{rf_xgb.mean()*100:.2f}%`")
        st.markdown(f"✅ **RF vs LGB Match**: `{rf_lgb.mean()*100:.2f}%`")
        st.markdown(f"✅ **LR vs XGB Match**: `{lr_xgb.mean()*100:.2f}%`")
        st.markdown(f"✅ **LR vs LGB Match**: `{lr_lgb.mean()*100:.2f}%`")
        st.markdown(f"✅ **XGB vs LGB Match**: `{xgb_lgb.mean()*100:.2f}%`")
        st.markdown(f"✅ **All 4 Match**: `{all_match.mean()*100:.2f}%`")

        comparison_df = pd.DataFrame({
            'Date': rf['Date'],
            'RF': rf['Prediction'],
            'LR': lr['Prediction'],
            'XGB': xgb['Prediction'],
            'LGB': lgb['Prediction'],
            'All Match': all_match
        })
        comparison_df[['RF', 'LR', 'XGB', 'LGB']] = comparison_df[['RF', 'LR', 'XGB', 'LGB']].replace({1: 'BUY', 0: 'HOLD', -1: 'SELL'})
        st.dataframe(comparison_df.tail(50).reset_index(drop=True), use_container_width=True)
    except Exception as e:
        st.error(f"Error in model comparison: {e}")

# --- Tab 8: XGBoost Report
with tabs[7]:
    st.subheader("🧪 XGBoost Results")
    if os.path.exists(report_path_xgb):
        st.markdown("**📊 Classification Report**")
        df = pd.read_csv(report_path_xgb, index_col=0)
        st.dataframe(df.style.format("{:.2f}"), use_container_width=True)
    if os.path.exists(signal_path_xgb):
        df = pd.read_csv(signal_path_xgb, parse_dates=['Date'])
        df['Prediction'] = df['Prediction'].replace({1: 'BUY', 0: 'HOLD', -1: 'SELL'})
        st.markdown("**📉 Signal Chart**")
        fig = px.scatter(df, x='Date', y='Prediction', color='Prediction',
                         color_discrete_map={'BUY': 'green', 'HOLD': 'blue', 'SELL': 'red'})
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df.tail(50).reset_index(drop=True), use_container_width=True)
    if os.path.exists(feature_importance_xgb):
        st.image(feature_importance_xgb, width=800)

# --- Tab 9: LightGBM Report
with tabs[8]:
    st.subheader("🧪 LightGBM Results")
    if os.path.exists(report_path_lgb):
        st.markdown("**📊 Classification Report**")
        df = pd.read_csv(report_path_lgb, index_col=0)
        st.dataframe(df.style.format("{:.2f}"), use_container_width=True)
    if os.path.exists(signal_path_lgb):
        df = pd.read_csv(signal_path_lgb, parse_dates=['Date'])
        df['Prediction'] = df['Prediction'].replace({1: 'BUY', 0: 'HOLD', -1: 'SELL'})
        st.markdown("**📉 Signal Chart**")
        fig = px.scatter(df, x='Date', y='Prediction', color='Prediction',
                         color_discrete_map={'BUY': 'green', 'HOLD': 'blue', 'SELL': 'red'})
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df.tail(50).reset_index(drop=True), use_container_width=True)
    if os.path.exists(feature_importance_lgb):
        st.image(feature_importance_lgb,  width=800)

# --- Sidebar
st.sidebar.title("📂 Project Files")
st.sidebar.markdown(f"[📁 Merged Dataset]({merged_dataset_path})")
st.sidebar.markdown(f"[📁 Unseen Dataset]({unseen_dataset_path})")
st.sidebar.markdown(f"[💾 RF Model]({model_path_rf})")
st.sidebar.markdown(f"[💾 LR Model]({model_path_lr})")
st.sidebar.markdown(f"[💾 XGB Model]({model_path_xgb})")
st.sidebar.markdown(f"[💾 LGBM Model]({model_path_lgb})")
st.sidebar.markdown(f"[🔗 Google Colab Code]({github_link})")

st.sidebar.info("Explore Forex sentiment predictions with ML models.")
