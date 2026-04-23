import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.preprocessing import load_and_merge, feature_engineering
from src.analysis import *
from src.model import train_model

st.set_page_config(page_title="Crypto Sentiment Analysis", layout="wide")

st.title("📊 Crypto Trader Sentiment Analysis")

# Upload files
trades_file = st.file_uploader("Upload Trades CSV", type=["csv"])
sentiment_file = st.file_uploader("Upload Fear & Greed CSV", type=["csv"])

if trades_file and sentiment_file:
    try:
        df = load_and_merge(trades_file, sentiment_file)
        df = feature_engineering(df)

        st.subheader("📌 Data Preview")
        st.dataframe(df.head())

        # Insights
        st.subheader("📊 Insights")

        col1, col2 = st.columns(2)

        with col1:
            st.write("### PnL by Sentiment")
            st.dataframe(sentiment_performance(df))

            st.write("### Win Rate")
            st.dataframe(win_rate(df))

        with col2:
            st.write("### Leverage")
            st.dataframe(leverage_analysis(df))

            st.write("### Trade Size")
            st.dataframe(size_analysis(df))

        # Visualization
        st.subheader("📉 PnL Distribution")

        fig, ax = plt.subplots()
        sns.boxplot(x='classification', y='closedpnl', data=df, ax=ax)
        st.pyplot(fig)

        # Risk
        st.subheader("⚠️ Leverage Risk")
        st.dataframe(leverage_risk(df))

        # Hourly
        st.subheader("⏰ Hourly Performance")
        st.dataframe(hourly_performance(df))

        # Model
        st.subheader("🤖 ML Model")

        model, acc = train_model(df)

        if model:
            st.write(f"Accuracy: {acc:.2f}")

            leverage = st.slider("Leverage", 1, 100, 10)
            size = st.number_input("Trade Size", value=100.0)

            if st.button("Predict"):
                pred = model.predict([[leverage, size]])
                result = "Profit ✅" if pred[0] else "Loss ❌"
                st.success(result)
        else:
            st.warning("Not enough data to train model")

    except Exception as e:
        st.error(f"Error: {str(e)}")