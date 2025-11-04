import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, SimpleRNN, Dropout, Dense
import os

# ---- PAGE CONFIG ----
st.set_page_config(page_title="Crypto Price Prediction", page_icon="💰", layout="wide")

st.title("💹 Cryptocurrency Price Prediction using RNN & LSTM")

# ---- FILE UPLOAD ----
uploaded_file = st.file_uploader("📂 Upload your Crypto CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Preview of Uploaded Data")
    st.write(df.head())

    # Auto-detect date and price columns
    date_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
    date_col = date_cols[0] if date_cols else df.columns[0]

    price_candidates = [c for c in df.columns if c.lower() in ('close', 'price', 'last', 'last_price', 'close_price')]
    price_col = price_candidates[0] if price_candidates else df.select_dtypes(include=[np.number]).columns[-1]

    df = df[[date_col, price_col]].dropna()
    df.columns = ['Date', 'Close']
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date')

    st.line_chart(df.set_index('Date')['Close'], height=300)

    # ---- DATA PREP ----
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(df[['Close']])

    seq_len = 60
    X, y = [], []
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i - seq_len:i])
        y.append(scaled[i])
    X, y = np.array(X), np.array(y)

    # ✅ Fix: Proper reshape (3D input for RNN/LSTM)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    st.write(f"✅ Data prepared: {X.shape[0]} sequences created.")

    # ---- MODEL CHOICE ----
    st.sidebar.header("🔧 Model Options")
    model_type = st.sidebar.selectbox("Choose model", ["LSTM", "SimpleRNN"])

    # ---- LOAD OR TRAIN MODEL ----
    model_path = f"{model_type.lower()}_model.h5"

    if os.path.exists(model_path):
        st.sidebar.success(f"Found saved {model_type} model — loading it...")
        model = load_model(model_path)
    else:
        st.sidebar.warning(f"No saved {model_type} model found — training a new one...")
        if model_type == "LSTM":
            model = Sequential([
                LSTM(64, input_shape=(seq_len, 1)),
                Dropout(0.2),
                Dense(1)
            ])
        else:
            model = Sequential([
                SimpleRNN(64, input_shape=(seq_len, 1)),
                Dropout(0.2),
                Dense(1)
            ])

        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=10, batch_size=32, verbose=1)
        model.save(model_path)
        st.sidebar.success(f"✅ {model_type} model trained and saved as {model_path}")

    # ---- PREDICT ----
    preds = model.predict(X)
    preds_inv = scaler.inverse_transform(preds)
    y_inv = scaler.inverse_transform(y)

    df_pred = df.iloc[seq_len:].copy()
    df_pred['Predicted'] = preds_inv

    st.subheader(f"📈 {model_type} Predictions vs True Values")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_pred['Date'], df_pred['Close'], label="True", linewidth=2)
    ax.plot(df_pred['Date'], df_pred['Predicted'], label="Predicted", alpha=0.8)
    ax.set_title(f"{model_type} Model — True vs Predicted Prices")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

    # ---- FUTURE FORECAST ----
    st.subheader("🔮 Future Price Forecast (Next 7 Days)")
    last_seq = scaled[-seq_len:]
    current_seq = last_seq.reshape(1, seq_len, 1)
    future_preds = []

    for _ in range(7):
        next_pred = model.predict(current_seq)[0, 0]
        future_preds.append(next_pred)
        current_seq = np.append(current_seq[:, 1:, :], [[[next_pred]]], axis=1)

    future_preds_inv = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))
    future_dates = pd.date_range(df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=7)
    df_future = pd.DataFrame({'Date': future_dates, 'Forecast': future_preds_inv.flatten()})

    st.write(df_future)
    st.line_chart(df_future.set_index('Date')['Forecast'], height=300)

    # ---- DOWNLOAD OPTION ----
    csv = df_future.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Forecast CSV", csv, "forecast.csv", "text/csv")
else:
    st.info("👆 Please upload a CSV file to start.")
