import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping

st.title("💹 Crypto Price Prediction using RNN and LSTM")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your crypto price CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Raw Data Preview", df.head())

    # Automatically detect Date and Close columns
    date_col = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
    price_col = [c for c in df.columns if 'close' in c.lower() or 'price' in c.lower()]
    date_col = date_col[0] if date_col else df.columns[0]
    price_col = price_col[0] if price_col else df.columns[-1]

    df = df[[date_col, price_col]].dropna()
    df[date_col] = pd.to_datetime(df[date_col], infer_datetime_format=True)
    df = df.sort_values(by=date_col)
    df.columns = ['Date','Close']
    df = df.set_index('Date')

    st.line_chart(df['Close'], use_container_width=True)

    seq_len = st.slider("Sequence length (days)", 10, 120, 60)
    epochs = st.slider("Epochs", 10, 200, 80)
    model_type = st.selectbox("Choose model type", ["RNN", "LSTM"])

    if st.button("🚀 Train Model"):
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df[['Close']].values)

        def create_sequences(values, seq_length):
            X, y = [], []
            for i in range(len(values) - seq_length):
                X.append(values[i:i+seq_length])
                y.append(values[i+seq_length])
            return np.array(X), np.array(y)

        train_ratio = 0.8
        train_size = int(len(scaled) * train_ratio)
        train_scaled = scaled[:train_size]
        test_scaled = scaled[train_size - seq_len:]
        X_train, y_train = create_sequences(train_scaled, seq_len)
        X_test, y_test = create_sequences(test_scaled, seq_len)
        X_train = X_train.reshape((X_train.shape[0], seq_le)
