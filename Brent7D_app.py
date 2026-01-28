import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go
import io
import warnings

warnings.filterwarnings("ignore")

# ─── Page Settings ──────────────────────────────────────────────────────────
st.set_page_config(page_title="Brent Forecast Pro", layout="wide")

# ─── Constants ──────────────────────────────────────────────────────────────
TICKER = "BZ=F"
ARIMA_ORDER = (5, 1, 0)
FORECAST_HORIZON = 7
BACKTEST_DAYS = 14


# ─── Data Loading Functions ─────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_data():
    data = yf.download(TICKER, period="2y", interval="1d", progress=False, auto_adjust=True)

    if data.empty:
        return pd.Series()

    if 'Close' in data.columns:
        prices = data['Close']
    else:
        prices = data.iloc[:, 0]

    if isinstance(prices, pd.DataFrame):
        prices = prices.iloc[:, 0]

    prices = prices.dropna()
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    return prices


# ─── Analysis Functions ─────────────────────────────────────────────────────
def run_backtest(prices, window=BACKTEST_DAYS):
    """Safe Walk-forward Backtest"""
    actual_window = min(window, len(prices) // 4)
    if actual_window < 5:
        return pd.Series(), pd.Series()

    preds = []
    actuals = []

    test_indices = range(actual_window, 0, -1)

    for i in test_indices:
        train = prices.iloc[:-i]
        actual = prices.iloc[-i]

        try:
            model = ARIMA(train, order=ARIMA_ORDER)
            model_fit = model.fit()
            forecast_val = model_fit.forecast(steps=1).iloc[-1]
            preds.append(forecast_val)
            actuals.append(actual)
        except:
            continue

    dates = prices.index[-len(preds):]
    return pd.Series(preds, index=dates), pd.Series(actuals, index=dates)


def get_forecast(prices, steps=FORECAST_HORIZON):
    """Future Forecast"""
    model = ARIMA(prices, order=ARIMA_ORDER)
    model_fit = model.fit()
    fc_values = model_fit.forecast(steps=steps)

    last_date = prices.index[-1]
    fc_index = pd.date_range(start=last_date + timedelta(days=1), periods=steps, freq='B')

    return pd.Series(fc_values.values, index=fc_index)


# ─── Main Interface ─────────────────────────────────────────────────────────
st.title("📊 Brent Crude Oil Price Forecast")

prices = load_data()

if prices.empty:
    st.error("Failed to load data from Yahoo Finance. Please refresh the page later.")
    st.stop()

if len(prices) < 30:
    st.warning(f"Not enough data for analysis (only {len(prices)} points found).")
    st.stop()

with st.spinner('Calculating forecast and metrics...'):
    # 1. Backtest
    backtest_preds, backtest_actuals = run_backtest(prices)

    # 2. Future Forecast
    forecast = get_forecast(prices)

# Sidebar for scenarios
st.sidebar.header("Settings")
scenario = st.sidebar.selectbox("Forecast Scenario:", ["Base Case", "Optimistic (+10%)", "Pessimistic (-10%)"])

if scenario == "Optimistic (+10%)":
    forecast = forecast * 1.10
elif scenario == "Pessimistic (-10%)":
    forecast = forecast * 0.90

# Metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Current Price", f"${prices.iloc[-1]:.2f}")

if not backtest_preds.empty:
    mae = mean_absolute_error(backtest_actuals, backtest_preds)
    mape = np.mean(np.abs((backtest_actuals - backtest_preds) / backtest_actuals)) * 100
    c2.metric("Accuracy (MAE)", f"${mae:.2f}")
    c4.metric("Error (MAPE)", f"{mape:.1f}%")
else:
    c2.write("Metrics unavailable")

# Tabs for Charts
tab1, tab2 = st.tabs(["📈 Forecast", "📉 Model Validation (Backtest)"])

with tab1:
    fig_fc = go.Figure()
    hist_tail = prices.tail(30)
    fig_fc.add_trace(
        go.Scatter(x=hist_tail.index, y=hist_tail.values, name="History", line=dict(color="#1f77b4", width=2)))
    fig_fc.add_trace(go.Scatter(x=forecast.index, y=forecast.values, name="Forecast",
                                line=dict(color="#ff7f0e", width=3, dash='dash')))
    # Connection line
    fig_fc.add_trace(
        go.Scatter(x=[hist_tail.index[-1], forecast.index[0]], y=[hist_tail.values[-1], forecast.values[0]],
                   showlegend=False, line=dict(color="#ff7f0e", dash='dash')))
    fig_fc.update_layout(title=f"Brent Crude: 7-Day Forecast ({scenario})", xaxis_title="Date", yaxis_title="USD",
                         hovermode="x unified")
    st.plotly_chart(fig_fc, use_container_width=True)

with tab2:
    if not backtest_preds.empty:
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(x=backtest_actuals.index, y=backtest_actuals.values, name="Actual Price"))
        fig_bt.add_trace(go.Scatter(x=backtest_preds.index, y=backtest_preds.values, name="Model Prediction"))
        fig_bt.update_layout(title="Walk-forward Backtest Results (Last 14 Days)", xaxis_title="Date",
                             yaxis_title="USD", hovermode="x unified")
        st.plotly_chart(fig_bt, use_container_width=True)
    else:
        st.write("Insufficient data for backtesting")

# Export Section
df_fc_export = pd.DataFrame({'Date': forecast.index.date, 'Forecast_USD': forecast.values.round(2)})
st.subheader("Forecast Data Table")
st.table(df_fc_export)

# Excel Generation
output = io.BytesIO()
with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
    df_fc_export.to_excel(writer, index=False, sheet_name='Forecast')
    if not backtest_preds.empty:
        metrics_df = pd.DataFrame({'Metric': ['MAE', 'MAPE (%)'], 'Value': [mae, mape]})
        metrics_df.to_excel(writer, index=False, sheet_name='Metrics')

st.download_button(
    label="📥 Download Excel Report",
    data=output.getvalue(),
    file_name=f"brent_forecast_{datetime.now().strftime('%Y%m%d')}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

st.caption(
    f"Data Source: Yahoo Finance (BZ=F). Model: ARIMA{ARIMA_ORDER}. Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
