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

# Отключаем лишние предупреждения
warnings.filterwarnings("ignore")

# ─── Настройки страницы ─────────────────────────────────────────────────────
st.set_page_config(page_title="Brent Forecast Pro", layout="wide")

# ─── Константы ───────────────────────────────────────────────────────────────
TICKER = "BZ=F"
ARIMA_ORDER = (5, 1, 0)
FORECAST_HORIZON = 7
BACKTEST_DAYS = 14
UPPER_ADJ = 1.10
LOWER_ADJ = 0.90


# ─── Функции загрузки данных ────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_data():
    # Загружаем за 2 года для обучения
    data = yf.download(TICKER, period="2y", progress=False)

    # Обработка MultiIndex (актуально для новых версий yfinance)
    if isinstance(data.columns, pd.MultiIndex):
        if 'Close' in data.columns.levels[0]:
            prices = data['Close'][TICKER]
        else:
            prices = data.iloc[:, 0]
    else:
        prices = data['Close']

    prices = prices.dropna()
    # Убираем часовой пояс, если он есть
    prices.index = prices.index.tz_localize(None)
    return prices


# ─── Функции анализа ────────────────────────────────────────────────────────
def run_backtest(prices, window=BACKTEST_DAYS):
    """Скользящий бэктест: предсказываем по 1 дню на основе прошлых данных"""
    preds = []
    actuals = []
    dates = prices.index[-window:]

    for i in range(window, 0, -1):
        # Обучающая выборка: всё, что было до i-го элемента с конца
        train = prices.iloc[:-i]
        actual = prices.iloc[-i]

        try:
            model = ARIMA(train, order=ARIMA_ORDER)
            model_fit = model.fit()
            # Исправлено: используем .iloc[0] чтобы избежать KeyError
            forecast_val = model_fit.forecast(steps=1).iloc[0]

            preds.append(forecast_val)
            actuals.append(actual)
        except:
            preds.append(np.nan)
            actuals.append(actual)

    return pd.Series(preds, index=dates), pd.Series(actuals, index=dates)


def get_forecast(prices, steps=FORECAST_HORIZON):
    """Прогноз на будущие 7 дней"""
    model = ARIMA(prices, order=ARIMA_ORDER)
    model_fit = model.fit()
    fc_values = model_fit.forecast(steps=steps)

    # Создаем индекс рабочих дней (Business days)
    last_date = prices.index[-1]
    fc_index = pd.date_range(start=last_date + timedelta(days=1), periods=steps, freq='B')

    return pd.Series(fc_values.values, index=fc_index)


# ─── Основной интерфейс Streamlit ──────────────────────────────────────────
st.title("📊 Прогноз цен Brent Crude")

with st.spinner('Загрузка данных и расчеты...'):
    prices = load_data()

    # 1. Запуск бэктеста
    backtest_preds, backtest_actuals = run_backtest(prices)

    # Расчет метрик
    mask = ~np.isnan(backtest_preds)
    y_true = backtest_actuals[mask]
    y_pred = backtest_preds[mask]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # 2. Получение будущего прогноза
    forecast = get_forecast(prices)

# Боковая панель для сценариев
st.sidebar.header("Параметры")
scenario = st.sidebar.selectbox(
    "Сценарий прогноза:",
    ["Базовый", "Геополитический риск (+10%)", "Спад спроса (-10%)"]
)

if scenario == "Геополитический риск (+10%)":
    forecast = forecast * UPPER_ADJ
elif scenario == "Спад спроса (-10%)":
    forecast = forecast * LOWER_ADJ

# Вывод метрик
c1, c2, c3, c4 = st.columns(4)
c1.metric("Текущая цена", f"${prices.iloc[-1]:.2f}")
c2.metric("Точность (MAE)", f"${mae:.2f}")
c3.metric("Ошибка (RMSE)", f"${rmse:.2f}")
c4.metric("Погрешность (MAPE)", f"{mape:.1f}%")

# Графики
tab1, tab2 = st.tabs(["📈 Прогноз", "📉 Проверка модели (Backtest)"])

with tab1:
    # График прогноза
    fig_fc = go.Figure()
    # Исторические данные (последние 30 дней)
    hist_tail = prices.tail(30)
    fig_fc.add_trace(
        go.Scatter(x=hist_tail.index, y=hist_tail.values, name="История", line=dict(color="#1f77b4", width=2)))
    # Прогноз
    fig_fc.add_trace(go.Scatter(x=forecast.index, y=forecast.values, name="Прогноз",
                                line=dict(color="#ff7f0e", width=3, dash='dash')))

    # Соединительная линия
    fig_fc.add_trace(go.Scatter(
        x=[hist_tail.index[-1], forecast.index[0]],
        y=[hist_tail.values[-1], forecast.values[0]],
        showlegend=False, line=dict(color="#ff7f0e", width=3, dash='dash')
    ))

    fig_fc.update_layout(title="Brent Crude: Прогноз на 7 дней", xaxis_title="Дата", yaxis_title="USD",
                         hovermode="x unified")
    st.plotly_chart(fig_fc, use_container_width=True)

with tab2:
    # График бэктеста
    fig_bt = go.Figure()
    fig_bt.add_trace(
        go.Scatter(x=backtest_actuals.index, y=backtest_actuals.values, name="Реальность", mode='lines+markers'))
    fig_bt.add_trace(
        go.Scatter(x=backtest_preds.index, y=backtest_preds.values, name="Предсказание модели", mode='lines+markers'))
    fig_bt.update_layout(title="Результаты Walk-forward Backtest (последние 14 дней)", xaxis_title="Дата",
                         yaxis_title="USD")
    st.plotly_chart(fig_bt, use_container_width=True)

# ─── Таблица и Excel ────────────────────────────────────────────────────────
st.subheader("Данные для экспорта")

# Сборка итоговой таблицы
df_hist_export = pd.DataFrame({'Цена': prices.tail(14), 'Тип': 'История'})
df_fc_export = pd.DataFrame({'Цена': forecast, 'Тип': f'Прогноз ({scenario})'})
final_df = pd.concat([df_hist_export, df_fc_export]).reset_index()
final_df.columns = ['Дата', 'Цена_USD', 'Тип_данных']
final_df['Дата'] = final_df['Дата'].dt.strftime('%Y-%m-%d')

st.dataframe(final_df, use_container_width=True)


# Функция генерации Excel
def convert_to_excel(df, mae, rmse, mape):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Forecast')
        # Лист с метриками
        metrics_df = pd.DataFrame({
            'Метрика': ['MAE', 'RMSE', 'MAPE (%)', 'Модель', 'Дата расчета'],
            'Значение': [mae, rmse, mape, 'ARIMA(5,1,0)', datetime.now().strftime('%Y-%m-%d %H:%M')]
        })
        metrics_df.to_excel(writer, index=False, sheet_name='Metrics')
    return output.getvalue()


excel_bytes = convert_to_excel(final_df, mae, rmse, mape)

st.download_button(
    label="💾 Скачать отчет в Excel",
    data=excel_bytes,
    file_name=f"brent_report_{datetime.now().strftime('%Y%m%d')}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

st.caption("Данные: Yahoo Finance (BZ=F). Метод: Walk-forward validation на 14 шагов.")
