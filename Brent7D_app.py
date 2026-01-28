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

# ─── Настройки страницы ─────────────────────────────────────────────────────
st.set_page_config(page_title="Brent Forecast Pro", layout="wide")

# ─── Константы ───────────────────────────────────────────────────────────────
TICKER = "BZ=F"
ARIMA_ORDER = (5, 1, 0)
FORECAST_HORIZON = 7
BACKTEST_DAYS = 14


# ─── Функции загрузки данных ────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_data():
    # Используем period="max" или "2y" и явно указываем auto_adjust
    data = yf.download(TICKER, period="2y", interval="1d", progress=False, auto_adjust=True)

    if data.empty:
        return pd.Series()

    # В новых версиях yfinance результат может быть MultiIndex или DataFrame
    # Берем колонку Close максимально надежным способом
    if 'Close' in data.columns:
        prices = data['Close']
    else:
        # Если колонок много (MultiIndex), берем первую доступную
        prices = data.iloc[:, 0]

    # Если это DataFrame (бывает при MultiIndex), превращаем в Series
    if isinstance(prices, pd.DataFrame):
        prices = prices.iloc[:, 0]

    prices = prices.dropna()
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    return prices


# ─── Функции анализа ────────────────────────────────────────────────────────
def run_backtest(prices, window=BACKTEST_DAYS):
    """Безопасный бэктест"""
    # Если данных слишком мало для бэктеста, уменьшаем окно
    actual_window = min(window, len(prices) // 4)
    if actual_window < 5:
        return pd.Series(), pd.Series()

    preds = []
    actuals = []

    # Берем последние n дней для теста
    test_indices = range(actual_window, 0, -1)

    for i in test_indices:
        train = prices.iloc[:-i]
        actual = prices.iloc[-i]

        try:
            model = ARIMA(train, order=ARIMA_ORDER)
            model_fit = model.fit()
            # Используем .iloc[-1] для получения последнего значения прогноза
            forecast_val = model_fit.forecast(steps=1).iloc[-1]
            preds.append(forecast_val)
            actuals.append(actual)
        except:
            continue

    dates = prices.index[-len(preds):]
    return pd.Series(preds, index=dates), pd.Series(actuals, index=dates)


def get_forecast(prices, steps=FORECAST_HORIZON):
    """Прогноз на будущее"""
    model = ARIMA(prices, order=ARIMA_ORDER)
    model_fit = model.fit()
    fc_values = model_fit.forecast(steps=steps)

    last_date = prices.index[-1]
    fc_index = pd.date_range(start=last_date + timedelta(days=1), periods=steps, freq='B')

    return pd.Series(fc_values.values, index=fc_index)


# ─── Основной интерфейс ─────────────────────────────────────────────────────
st.title("📊 Прогноз цен Brent Crude")

prices = load_data()

if prices.empty:
    st.error("Не удалось загрузить данные из Yahoo Finance. Попробуйте обновить страницу позже.")
    st.stop()

# Проверка на достаточное количество данных
if len(prices) < 30:
    st.warning(f"Слишком мало данных для анализа (всего {len(prices)} точек).")
    st.stop()

with st.spinner('Выполняем расчеты...'):
    # 1. Бэктест
    backtest_preds, backtest_actuals = run_backtest(prices)

    # 2. Будущий прогноз
    forecast = get_forecast(prices)

# Сценарии в сайдбаре
st.sidebar.header("Параметры")
scenario = st.sidebar.selectbox("Сценарий:", ["Базовый", "Оптимистичный (+10%)", "Пессимистичный (-10%)"])

if scenario == "Оптимистичный (+10%)":
    forecast = forecast * 1.10
elif scenario == "Пессимистичный (-10%)":
    forecast = forecast * 0.90

# Метрики (показываем только если бэктест удался)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Текущая цена", f"${prices.iloc[-1]:.2f}")

if not backtest_preds.empty:
    mae = mean_absolute_error(backtest_actuals, backtest_preds)
    mape = np.mean(np.abs((backtest_actuals - backtest_preds) / backtest_actuals)) * 100
    c2.metric("Точность (MAE)", f"${mae:.2f}")
    c4.metric("Погрешность (MAPE)", f"{mape:.1f}%")
else:
    c2.write("Метрики недоступны")

# Графики
tab1, tab2 = st.tabs(["📈 Прогноз", "📉 Проверка модели"])

with tab1:
    fig_fc = go.Figure()
    hist_tail = prices.tail(30)
    fig_fc.add_trace(
        go.Scatter(x=hist_tail.index, y=hist_tail.values, name="История", line=dict(color="#1f77b4", width=2)))
    fig_fc.add_trace(go.Scatter(x=forecast.index, y=forecast.values, name="Прогноз",
                                line=dict(color="#ff7f0e", width=3, dash='dash')))
    # Линия соединения
    fig_fc.add_trace(
        go.Scatter(x=[hist_tail.index[-1], forecast.index[0]], y=[hist_tail.values[-1], forecast.values[0]],
                   showlegend=False, line=dict(color="#ff7f0e", dash='dash')))
    fig_fc.update_layout(title="Brent Crude: Прогноз на 7 дней", xaxis_title="Дата", yaxis_title="USD")
    st.plotly_chart(fig_fc, use_container_width=True)

with tab2:
    if not backtest_preds.empty:
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(x=backtest_actuals.index, y=backtest_actuals.values, name="Факт"))
        fig_bt.add_trace(go.Scatter(x=backtest_preds.index, y=backtest_preds.values, name="Модель"))
        fig_bt.update_layout(title="Результаты бэктеста", xaxis_title="Дата", yaxis_title="USD")
        st.plotly_chart(fig_bt, use_container_width=True)
    else:
        st.write("Недостаточно данных для бэктеста")

# Экспорт
df_fc_export = pd.DataFrame({'Дата': forecast.index.date, 'Прогноз_USD': forecast.values.round(2)})
st.subheader("Таблица прогноза")
st.table(df_fc_export)

# Генерация Excel
output = io.BytesIO()
with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
    df_fc_export.to_excel(writer, index=False, sheet_name='Forecast')
st.download_button(label="📥 Скачать в Excel", data=output.getvalue(), file_name="brent_forecast.xlsx")
