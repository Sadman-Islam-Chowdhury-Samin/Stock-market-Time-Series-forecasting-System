import os
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import gradio as gr
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load your pre-trained LSTM model
lstm_model = tf.keras.models.load_model("lstm_model.h5", compile=False)

# Load dataset CSV (world stock prices from Kaggle)
df = pd.read_csv("world_stock_prices.csv", parse_dates=["Date"])

# Forecast helper (from your code)
def forecast_iterative_lstm(model, last_window_scaled, fh=5):
    seq_len = len(last_window_scaled)
    preds = []
    window = last_window_scaled.copy().reshape(-1)
    for _ in range(fh):
        x = window.reshape(1, seq_len, 1)
        yhat = model.predict(x, verbose=0)[0,0]
        preds.append(yhat)
        window = np.append(window[1:], yhat)
    return np.array(preds)

# Forecast function for Gradio
def forecast_stock(ticker, horizon):
    # Filter ticker data
    data = df[df["Ticker"] == ticker].sort_values("Date")
    if data.empty:
        return {"error": "Ticker not found in dataset."}, None

    ts = data["Close"].values
    dates = data["Date"].values

    seq_len = 60
    if len(ts) < seq_len:
        return {"error": f"Not enough data for forecasting (need at least {seq_len} days)."}, None

    # Fit MinMaxScaler on last seq_len points (per-window scaling)
    last_window = ts[-seq_len:].reshape(-1, 1)
    scaler_local = MinMaxScaler()
    last_window_scaled = scaler_local.fit_transform(last_window).flatten()

    # Forecast
    preds_scaled = forecast_iterative_lstm(lstm_model, last_window_scaled, fh=horizon)
    preds = scaler_local.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()

    # Create forecast dates
    last_date = pd.to_datetime(dates[-1])
    forecast_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=horizon)

    forecast_series = pd.Series(preds, index=forecast_dates)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(pd.to_datetime(dates), ts, label="Historical Data")
    plt.plot(forecast_series.index, forecast_series.values, label="Forecast", linestyle='--', marker='o')
    plt.axvline(pd.to_datetime(dates[-1]), color='k', linestyle='--', alpha=0.5)
    plt.title(f"LSTM Forecast for {ticker}")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.tight_layout()

    return forecast_series.to_dict(), plt.gcf()

# Gradio interface
iface = gr.Interface(
    fn=forecast_stock,
    inputs=[
        gr.Textbox(label="Stock Ticker", placeholder="e.g., AAPL"),
        gr.Slider(1, 30, step=1, value=5, label="Forecast Horizon (days)")
    ],
    outputs=[
        gr.JSON(label="Forecast Values"),
        gr.Plot(label="Forecast Plot")
    ],
    title="📈 Stock Forecasting with LSTM",
    description="Enter a ticker from the Kaggle dataset and forecast future prices using your LSTM model."
)

if __name__ == "__main__":
    iface.launch()
