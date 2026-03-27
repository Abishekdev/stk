import datetime
import time

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers, models
import plotly.graph_objects as go
import streamlit as st


# -----------------------------
# Config
# -----------------------------
SEQUENCE_LENGTH = 30
TEST_SPLIT = 0.2
EPOCHS = 5
BATCH_SIZE = 32
DEFAULT_START_DATE = "2015-01-01"
USD_TO_INR = 95.00 # Current exchange rate (update as needed)


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
    # Set page config for professional look
    st.set_page_config(
        page_title="Stock Price Predictor",
        page_icon="�",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS for professional styling
    st.markdown("""
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        <style>
        /* Main title styling */
        .main-title {
            font-size: 2.8rem;
            font-weight: 700;
            color: #1f77b4;
            margin-bottom: 0.5rem;
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        .subtitle {
            font-size: 1.1rem;
            color: #666;
            margin-bottom: 2rem;
        }
        /* Card styling */
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 10px;
            color: white;
            margin-bottom: 1rem;
        }
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
        }
        .metric-label {
            font-size: 0.9rem;
            opacity: 0.9;
        }
        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: 600;
            border: none;
            padding: 0.75rem 2rem;
            border-radius: 8px;
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 16px rgba(102, 126, 234, 0.4);
        }
        /* Sidebar styling */
        .css-1d391kg {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        /* Icon styling */
        .icon-text {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .section-header {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            font-weight: bold;
            font-size: 1.8rem;
            color: #1f77b4;
            margin: 2rem 0 1.5rem 0;
            padding-bottom: 0.75rem;
            border-bottom: 3px solid #667eea;
        }
        .section-header i {
            font-size: 2rem;
            color: #667eea;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<div class="main-title"><i class="fas fa-chart-line"></i> Stock Price Predictor</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">AI-Powered Next-Day Price Forecasting with LSTM Neural Networks</div>', unsafe_allow_html=True)
    
    with col2:
        if st.button("Clear Cache", key="clear_cache"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.session_state.pop("model_cache", None)
            st.success("Cache cleared successfully!")

    st.divider()

    # Input Section
    st.markdown('<div class="section-header"><i class="fas fa-sliders-h"></i> Configuration</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        ticker = st.text_input("Stock Ticker", value="AAPL", help="Enter the stock symbol (e.g., AAPL, GOOGL, MSFT)")
    
    with col2:
        start_date = st.date_input(
            "Start Date",
            value=datetime.date.fromisoformat(DEFAULT_START_DATE),
            help="Select the beginning of your training data period"
        )
    
    with col3:
        use_today = st.checkbox("Use Today", value=True, help="Use today's date as the end date")
    
    with col4:
        if use_today:
            end_date = None
            st.write("End: Today")
        else:
            end_date = st.date_input("End Date", value=datetime.date.today())

    col1, col2 = st.columns(2)
    
    with col1:
        epochs = st.slider(
            "Training Epochs",
            min_value=1,
            max_value=10,
            value=EPOCHS,
            step=1,
            help="Number of times the model will train on the data (higher = more accurate but slower)"
        )
    
    with col2:
        st.info("More epochs generally improve accuracy but take longer to train.")

    st.divider()

    # Prediction Button
    col1, col2, col3 = st.columns([1, 1, 2])
    with col2:
        predict_button = st.button("Run Prediction", key="predict", use_container_width=True)

    if predict_button:
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

            st.success("Prediction completed successfully!")
            st.divider()

            # Results Section
            st.markdown('<div class="section-header"><i class="fas fa-chart-bar"></i> Results</div>', unsafe_allow_html=True)

            # Key Metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                price_inr = next_day_price * USD_TO_INR
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Next-Day Predicted Price</div>
                        <div class="metric-value">${next_day_price:.2f}</div>
                        <div style="font-size: 1.3rem; margin-top: 0.5rem;">₹{price_inr:.2f}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Training Time</div>
                        <div class="metric-value">{train_time:.2f}s</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col3:
                cache_status = "Cache Hit" if cache_hit else "Fresh Train"
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Status</div>
                        <div class="metric-value">{cache_status}</div>
                    </div>
                """, unsafe_allow_html=True)

            st.divider()

            # Chart Section
            st.markdown('<div class="section-header"><i class="fas fa-chart-area"></i> Actual vs Predicted (Test Data)</div>', unsafe_allow_html=True)
            
            # Create interactive Plotly chart
            fig = go.Figure()
            
            # Add actual price line with gradient fill
            fig.add_trace(go.Scatter(
                x=list(range(len(y_test_actual))),
                y=y_test_actual.flatten(),
                mode='lines+markers',
                name='Actual Price',
                line=dict(color='#9D4EDD', width=4),
                marker=dict(
                    size=6,
                    color='#9D4EDD',
                    symbol='circle',
                    line=dict(color='white', width=1)
                ),
                hovertemplate=(
                    '<b style="font-size:14px">Actual Price</b><br>' +
                    '<b>Data Point:</b> %{x}<br>' +
                    '<b>Price:</b> <span style="color:#9D4EDD">$%{y:.2f}</span><br>' +
                    '<extra></extra>'
                ),
                fill='tozeroy',
                fillcolor='rgba(157, 78, 221, 0.3)'
            ))
            
            # Add predicted price line
            fig.add_trace(go.Scatter(
                x=list(range(len(y_pred))),
                y=y_pred.flatten(),
                mode='lines+markers',
                name='Predicted Price',
                line=dict(color='#FFD60A', width=4, dash='solid'),
                marker=dict(
                    size=5,
                    color='#FFD60A',
                    symbol='diamond',
                    line=dict(color='white', width=1)
                ),
                hovertemplate=(
                    '<b style="font-size:14px">Predicted Price</b><br>' +
                    '<b>Data Point:</b> %{x}<br>' +
                    '<b>Price:</b> <span style="color:#FFD60A">$%{y:.2f}</span><br>' +
                    '<extra></extra>'
                ),
                fill=None
            ))
            
            # Update layout for dark theme with enhanced interactivity
            fig.update_layout(
                title={
                    'text': f'<b style="font-size:22px">{ticker.upper()} Close Price Prediction Analysis</b><br>' +
                            '<sub style="font-size:12px; color:#999">Interactive chart - Hover, Zoom, Pan, and Select time periods</sub>',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20, 'color': '#FFFFFF', 'family': 'Arial Black'}
                },
                xaxis_title='<b style="color:#DDD">Test Data Points</b>',
                yaxis_title='<b style="color:#DDD">Price ($)</b>',
                hovermode='x unified',
                plot_bgcolor='#1a1a2e',
                paper_bgcolor='#16213e',
                font=dict(family='Arial, sans-serif', size=12, color='#DDD'),
                legend=dict(
                    x=0.02,
                    y=0.98,
                    bgcolor='rgba(26, 26, 46, 0.95)',
                    bordercolor='#9D4EDD',
                    borderwidth=2,
                    font=dict(size=13, color='#DDD'),
                    title=dict(text='<b>Legend</b>', font=dict(size=14, color='#9D4EDD'))
                ),
                xaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(157, 78, 221, 0.2)',
                    zeroline=False,
                    showline=True,
                    linewidth=2,
                    linecolor='#9D4EDD'
                ),
                yaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(157, 78, 221, 0.2)',
                    zeroline=False,
                    showline=True,
                    linewidth=2,
                    linecolor='#9D4EDD'
                ),
                margin=dict(l=80, r=40, t=120, b=80),
                height=650,
                dragmode='zoom'
            )
            
            # Add range slider and enhanced buttons
            fig.update_xaxes(
                rangeslider_visible=True,
                rangeslider_thickness=0.05,
                rangeselector=dict(
                    buttons=list([
                        dict(
                            count=5,
                            label='First 5',
                            step='day',
                            stepmode='backward'
                        ),
                        dict(
                            count=10,
                            label='First 10',
                            step='day',
                            stepmode='backward'
                        ),
                        dict(
                            count=1,
                            label='First 30%',
                            step='day',
                            stepmode='backward'
                        ),
                        dict(
                            step='all',
                            label='All Data'
                        )
                    ]),
                    bgcolor='#667eea',
                    activecolor='#764ba2',
                    font=dict(color='white', size=11, family='Arial'),
                    x=0,
                    y=1.15
                )
            )
            
            # Add annotations for better UX
            fig.add_annotation(
                text='Tip: Use the slider below to zoom into specific time periods. Hover over points for exact values.',
                xref='paper',
                yref='paper',
                x=0.5,
                y=-0.15,
                showarrow=False,
                font=dict(size=11, color='#666', family='Arial'),
                xanchor='center'
            )
            
            st.plotly_chart(fig, use_container_width=True, config={
                'responsive': True,
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': f'{ticker.upper()}_prediction',
                    'height': 700,
                    'width': 1200,
                    'scale': 2
                }
            })

            # Detailed Statistics
            st.markdown('<div class="section-header"><i class="fas fa-calculator"></i> Detailed Statistics</div>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                mse = np.mean((y_test_actual - y_pred) ** 2)
                st.metric("Mean Squared Error", f"{mse:.4f}")
            
            with col2:
                mae = np.mean(np.abs(y_test_actual - y_pred))
                mae_inr = mae * USD_TO_INR
                st.metric("Mean Absolute Error", f"${mae:.2f} (₹{mae_inr:.2f})")
            
            with col3:
                rmse = np.sqrt(mse)
                st.metric("Root Mean Squared Error", f"${rmse:.2f}")

        except Exception as exc:
            st.error(f"Error: {exc}")
            st.info("Please check your inputs and try again.")


if __name__ == "__main__":
    main()
