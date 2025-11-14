"""
app.py — Stock Price Forecasting with LSTM
For Hugging Face Spaces deployment.

Required files in repo root:
 - lstm_final.h5
 - scaler.pkl

Optional:
 - Set HF_MODEL_REPO=<username/repo> in Space secrets if model files are hosted separately.
"""

import os
import io
import shutil
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr

from tensorflow.keras.models import load_model

# for downloading model files from HF Hub
try:
    from huggingface_hub import hf_hub_download
    HF_HUB_AVAILABLE = True
except Exception:
    HF_HUB_AVAILABLE = False


# --------- Download model files from HF Hub if needed ----------
def ensure_model_file(fname: str, repo_env: str = "HF_MODEL_REPO"):
    """Download model file from HF Hub if not found locally."""
    if os.path.exists(fname):
        return True
    repo = os.environ.get(repo_env)
    if repo and HF_HUB_AVAILABLE:
        try:
            print(f"Downloading {fname} from repo {repo}...")
            path = hf_hub_download(repo_id=repo, filename=fname)
            shutil.copy(path, fname)
            return True
        except Exception as e:
            print(f"Download failed for {fname}: {e}")
    return False


# --------- Load LSTM model & scaler (cached) ----------
_model_cache = {"lstm": None, "scaler": None}

def load_lstm_and_scaler(model_path="lstm_final.h5", scaler_path="scaler.pkl"):
    ensure_model_file(model_path)
    ensure_model_file(scaler_path)

    if _model_cache["lstm"] is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing model file: {model_path}")
        _model_cache["lstm"] = load_model(model_path, compile=False)
        print("✅ LSTM model loaded.")
    if _model_cache["scaler"] is None:
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Missing scaler file: {scaler_path}")
        _model_cache["scaler"] = joblib.load(scaler_path)
        print("✅ Scaler loaded.")
    return _model_cache["lstm"], _model_cache["scaler"]


# --------- Input parsing ----------
def parse_input(file_obj, paste_text, ticker):
    """
    Priority:
     1) Uploaded CSV file
     2) Pasted comma-separated values
     3) Ticker (via yfinance)
    Returns: pd.Series, pd.DatetimeIndex or None
    """
    # 1. CSV upload
    if file_obj is not None:
        df = pd.read_csv(file_obj.name) if hasattr(file_obj, "name") else pd.read_csv(io.StringIO(file_obj.read().decode()))
        series = _find_price_series(df)
        date_idx = _find_date_index(df)
        return series, date_idx

    # 2. Pasted values
    if paste_text:
        values = [float(x.strip()) for x in paste_text.replace("\n", ",").split(",") if x.strip()]
        return pd.Series(values), None

    # 3. Ticker
    if ticker:
        import yfinance as yf
        data = yf.download(ticker, progress=False)
        if data is None or data.empty:
            raise ValueError(f"No data found for ticker: {ticker}")
        return data["Close"].reset_index(drop=True), pd.to_datetime(data.index)

    raise ValueError("Please provide input: CSV, pasted values, or ticker.")


def _find_price_series(df: pd.DataFrame) -> pd.Series:
    candidates = ["Close", "close", "Adj Close", "adjclose", "Adj_Close", "price"]
    for col in candidates:
        if col in df.columns:
            return df[col].astype(float).reset_index(drop=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        return df[numeric_cols[0]].astype(float).reset_index(drop=True)
    raise ValueError("No numeric price column found in CSV.")


def _find_date_index(df: pd.DataFrame):
    for col in ["Date", "date", "DATE", "Timestamp", "timestamp"]:
        if col in df.columns:
            try:
                return pd.to_datetime(df[col])
            except Exception:
                pass
    return None


# --------- Forecasting ----------
def pad_sequence(seq: np.ndarray, seq_len: int):
    """Pad short sequences by repeating last value."""
    if len(seq) >= seq_len:
        return seq
    pad_len = seq_len - len(seq)
    return np.concatenate([seq, np.repeat(seq[-1], pad_len)])


def forecast_lstm(model, scaler, series, horizon=5, seq_len=60):
    data = np.array(series).astype(float).reshape(-1, 1)
    data = pad_sequence(data.flatten(), seq_len).reshape(-1, 1)
    scaled = scaler.transform(data).flatten().tolist()

    predictions_scaled = []
    for _ in range(horizon):
        x_input = np.array(scaled[-seq_len:]).reshape(1, seq_len, 1)
        yhat = model.predict(x_input, verbose=0)
        yhat = float(np.array(yhat).flatten()[0])
        predictions_scaled.append([yhat])
        scaled.append(yhat)

    preds = scaler.inverse_transform(np.array(predictions_scaled))
    return preds.flatten().tolist()


def plot_forecast(history_vals, history_idx, preds, preds_idx):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(history_idx if history_idx is not None else range(len(history_vals)),
            history_vals, label="History")
    ax.plot(preds_idx, preds, label="Forecast", linestyle="--", marker="o")
    ax.set_title("LSTM Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    plt.tight_layout()
    return fig


# --------- Gradio function ----------
def run_forecast(file, paste_text, ticker, horizon, seq_len):
    try:
        series, date_idx = parse_input(file, paste_text, ticker)
    except Exception as e:
        return f"❌ Error: {e}", None, None

    try:
        model, scaler = load_lstm_and_scaler()
        preds = forecast_lstm(model, scaler, series, int(horizon), int(seq_len))
    except Exception as e:
        return f"❌ Forecasting failed: {e}", None, None

    last_date = date_idx.iloc[-1] if date_idx is not None else pd.Timestamp.today()
    future_idx = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=int(horizon))

    forecast_df = pd.DataFrame({"date": future_idx, "forecast": preds})
    fig = plot_forecast(series, date_idx, preds, future_idx)

    return "✅ Forecast generated successfully", fig, forecast_df


# --------- Gradio UI ----------
title = "📈 LSTM Stock Price Forecasting"
description = """
Upload a CSV (e.g., from the [World Stock Prices](https://www.kaggle.com/datasets/nelgiriyewithana/world-stock-prices-daily-updating) dataset),
paste comma-separated values, or provide a ticker (via yfinance).
Make sure `lstm_final.h5` and `scaler.pkl` are in the repo root.
"""

with gr.Blocks(title=title) as demo:
    gr.Markdown(f"## {title}")
    gr.Markdown(description)

    with gr.Row():
        with gr.Column(scale=2):
            file_in = gr.File(label="Upload CSV")
            paste_in = gr.Textbox(label="Or paste comma-separated closing prices", lines=3)
            ticker_in = gr.Textbox(label="Or provide ticker (optional)", placeholder="e.g. AAPL")
            horizon_in = gr.Number(value=5, label="Forecast Horizon (days)", precision=0)
            seq_in = gr.Number(value=60, label="LSTM Sequence Length", precision=0)
            run_btn = gr.Button("🔮 Run Forecast")
            out_msg = gr.Textbox(label="Status")
        with gr.Column(scale=3):
            out_plot = gr.Plot(label="Forecast Plot")
            out_table = gr.DataFrame(label="Forecast Results")

    run_btn.click(run_forecast, [file_in, paste_in, ticker_in, horizon_in, seq_in],
                  [out_msg, out_plot, out_table])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
