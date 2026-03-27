import datetime
import time

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import streamlit as st


# -----------------------------
# Config
# -----------------------------
SEQUENCE_LENGTH = 30
TEST_SPLIT = 0.2
EPOCHS = 5
BATCH_SIZE = 32
DEFAULT_START_DATE = "2015-01-01"


@st.cache_data(show_spinner=False)
def fetch_close_prices(ticker, start_date, end_date=None):
    """Fetch historical close prices from Yahoo Finance."""
    end = end_date or datetime.date.today().isoformat()
    data = yf.download(ticker, start=start_date, end=end, progress=False)
    if data.empty:
        raise ValueError("No data returned. Check ticker or date range.")
    return data[["Close"]].copy()


def create_sequences(values, sequence_length):
    """Create sequences of past days to predict the next day."""
    x, y = [], []
    for i in range(sequence_length, len(values)):
        x.append(values[i - sequence_length : i])
        y.append(values[i])
    return np.array(x), np.array(y)


def build_model(sequence_length):
    """Build a simple LSTM model."""
    model = models.Sequential(
        [
            layers.Input(shape=(sequence_length, 1)),
            layers.LSTM(50),
            layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


@st.cache_resource(show_spinner=False)
def train_model_cached(ticker, start_date, end_date, epochs):
    """Train and cache model per ticker/date range for faster repeats."""
    close_df = fetch_close_prices(ticker, start_date, end_date)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_values = scaler.fit_transform(close_df.values)

    x_all, y_all = create_sequences(scaled_values, SEQUENCE_LENGTH)

    split_index = int(len(x_all) * (1 - TEST_SPLIT))
    x_train, x_test = x_all[:split_index], x_all[split_index:]
    y_train, y_test = y_all[:split_index], y_all[split_index:]

    model = build_model(SEQUENCE_LENGTH)
    model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        validation_data=(x_test, y_test),
        verbose=0,
    )

    return model, scaler, scaled_values, x_test, y_test


def run_prediction(ticker, start_date, end_date=None, epochs=EPOCHS):
    cache_key = (ticker.upper(), start_date, end_date, epochs)
    model_cache = st.session_state.setdefault("model_cache", {})

    if cache_key in model_cache:
        model, scaler, scaled_values, x_test, y_test, train_time = model_cache[
            cache_key
        ]
        cache_hit = True
    else:
        start_time = time.perf_counter()
        model, scaler, scaled_values, x_test, y_test = train_model_cached(
            ticker,
            start_date,
            end_date,
            epochs,
        )
        train_time = time.perf_counter() - start_time
        model_cache[cache_key] = (
            model,
            scaler,
            scaled_values,
            x_test,
            y_test,
            train_time,
        )
        cache_hit = False

    # Predict on test data
    y_pred_scaled = model.predict(x_test)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_test_actual = scaler.inverse_transform(y_test)

    # Predict next day
    last_sequence = scaled_values[-SEQUENCE_LENGTH:]
    next_day_scaled = model.predict(last_sequence.reshape(1, SEQUENCE_LENGTH, 1))
    next_day_price = scaler.inverse_transform(next_day_scaled)[0][0]

    return y_test_actual, y_pred, next_day_price, train_time, cache_hit


def main():
    st.title("Simple Stock Price Predictor")
    st.write("Predict next-day stock prices with a basic LSTM model.")

    ticker = st.text_input("Stock ticker", value="AAPL")
    start_date = st.date_input(
        "Start date",
        value=datetime.date.fromisoformat(DEFAULT_START_DATE),
    )
    use_today = st.checkbox("Use today as end date", value=True)
    end_date = None
    if not use_today:
        end_date = st.date_input("End date", value=datetime.date.today())
    epochs = st.slider("Epochs", min_value=1, max_value=10, value=EPOCHS, step=1)

    if st.button("Clear cache"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.session_state.pop("model_cache", None)
        st.success("Cache cleared.")

    if st.button("Run Prediction"):
        try:
            with st.spinner("Fetching data and training model..."):
                y_test_actual, y_pred, next_day_price, train_time, cache_hit = (
                    run_prediction(
                        ticker,
                        start_date.isoformat(),
                        end_date.isoformat() if end_date else None,
                        epochs,
                    )
                )

            st.subheader("Training Details")
            st.write(f"Cache hit: {'Yes' if cache_hit else 'No'}")
            st.write(f"Training time: {train_time:.2f} seconds")

            st.subheader("Next-Day Predicted Close Price")
            st.write(f"{ticker.upper()}: {next_day_price:.2f}")

            st.subheader("Actual vs Predicted (Test Data)")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(y_test_actual, label="Actual")
            ax.plot(y_pred, label="Predicted")
            ax.set_title(f"{ticker.upper()} Close Price Prediction")
            ax.set_xlabel("Test Data Points")
            ax.set_ylabel("Price")
            ax.legend()
            st.pyplot(fig)

        except Exception as exc:
            st.error(f"Error: {exc}")


if __name__ == "__main__":
    main()
