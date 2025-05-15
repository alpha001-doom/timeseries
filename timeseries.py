# --- Install dependencies only in Google Colab ---
import sys
if 'google.colab' in sys.modules:
    !pip install yfinance prophet ipywidgets statsmodels tensorflow --quiet
    from google.colab import output
    output.enable_custom_widget_manager()

# --- Imports ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import timedelta, datetime
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
import warnings
warnings.filterwarnings("ignore")

# --- UI Widgets ---
popular_tickers = ['AAPL', 'MSFT', 'TSLA', 'GOOGL', 'AMZN', 'NVDA', 'META', 'NFLX']

ticker_dropdown = widgets.Dropdown(
    options=popular_tickers,
    value='AAPL',
    description='Ticker:'
)

custom_ticker = widgets.Text(
    placeholder='Type your own (e.g., INFY.NS)',
    description='Custom:'
)

start_date = widgets.DatePicker(
    description='Start:',
    value=pd.to_datetime('2018-01-01')
)

forecast_days = widgets.IntSlider(
    value=1,
    min=1,
    max=5,
    step=1,
    description='Days:'
)

run_btn = widgets.Button(description='📈 Run Forecast', button_style='success')
clear_btn = widgets.Button(description='🧹 Clear Output', button_style='warning')

export_check = widgets.Checkbox(
    value=False,
    description='Export as CSV'
)

ui_layout = widgets.VBox([
    widgets.HTML("<h2 style='color:#2c3e50;'>📊 Stock Forecast Dashboard</h2>"),
    widgets.HBox([ticker_dropdown, custom_ticker]),
    widgets.HBox([start_date, forecast_days]),
    widgets.HBox([run_btn, clear_btn, export_check]),
    widgets.HTML("<hr>")
])

output_area = widgets.Output()
display(ui_layout, output_area)

# --- Helper UI Card ---
def show_summary_card(title, value, change=None, color='black'):
    arrow = ''
    if change is not None:
        arrow = '🔺' if change > 0 else '🔻'
    return f"""
        <div style="display:inline-block; width:200px; margin:10px; padding:10px; border-radius:10px; background:#f7f7f7; text-align:center;">
            <h4 style="margin:0;">{title}</h4>
            <p style="color:{color}; font-size:20px; margin:5px 0;">{arrow} ${value:.2f}</p>
        </div>
    """

# --- Main Forecast Function ---
def run_forecast(ticker, start, days):
    with output_area:
        clear_output()
        print(f"Fetching data for {ticker}...")

        try:
            df = yf.download(ticker, start=start.strftime('%Y-%m-%d'), end=datetime.today().strftime('%Y-%m-%d'))
            df = df[['Close']].dropna().reset_index()
        except Exception as e:
            print(f"Data fetch error: {e}")
            return

        if len(df) < 100:
            print("Not enough data. Try an earlier start date.")
            return

        display(HTML("<p><b>Running models...</b> ⏳</p>"))
        start_time = datetime.now()

        # --- LSTM Model ---
        def lstm_model(df):
            data = df[['Close']].values
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(data)
            X, y = [], []
            for i in range(60, len(scaled)):
                X.append(scaled[i-60:i])
                y.append(scaled[i])
            X, y = np.array(X), np.array(y)
            X = X.reshape(X.shape[0], X.shape[1], 1)

            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
                Dropout(0.2),
                LSTM(50),
                Dropout(0.2),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X, y, epochs=10, batch_size=32, verbose=0)

            forecast = []
            last_input = scaled[-60:]
            for _ in range(days):
                pred_scaled = model.predict(last_input.reshape(1, 60, 1), verbose=0)
                forecast.append(pred_scaled[0][0])
                last_input = np.append(last_input[1:], [[pred_scaled[0][0]]], axis=0)
            forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()
            return forecast

        # --- ARIMA Model ---
        def arima_model(df):
            model = ARIMA(df['Close'], order=(5, 1, 0))
            fit = model.fit()
            return fit.forecast(steps=days).values

        # --- Prophet Model ---
        def prophet_model(df):
            p_df = df[['Date', 'Close']].copy()
            p_df.columns = ['ds', 'y']
            model = Prophet(daily_seasonality=True)
            model.fit(p_df)
            future = model.make_future_dataframe(periods=days)
            forecast = model.predict(future)
            return forecast.iloc[-days:]['yhat'].values

        try:
            lstm = lstm_model(df)
            arima = arima_model(df)
            prophet = prophet_model(df)
            ensemble = (lstm + arima + prophet) / 3
            runtime = (datetime.now() - start_time).total_seconds()
        except Exception as e:
            print(f"Model error: {e}")
            return

        last_price = df['Close'].iloc[-1].item()
        last_date = df['Date'].iloc[-1]
        future_dates = [last_date + timedelta(days=i) for i in range(1, days + 1)]

        results = pd.DataFrame({
            'Date': [d.strftime('%Y-%m-%d') for d in future_dates],
            'LSTM': lstm.round(2),
            'ARIMA': arima.round(2),
            'Prophet': prophet.round(2),
            'Ensemble': ensemble.round(2)
        })

        # --- Display Table (fixed style without KeyError) ---
        styled_results = results.style.set_table_styles(
            [{'selector': 'th', 'props': [('text-align', 'center')]}]
        ).set_properties(**{'text-align': 'center'})

        display(HTML(f"<h4>📅 Forecast Results for <code>{ticker}</code></h4>"))
        display(styled_results)

        # --- Summary Cards ---
        change = ((ensemble[0] - last_price) / last_price) * 100
        html_cards = (
            show_summary_card("Last Close", last_price) +
            show_summary_card("Forecast (D1)", ensemble[0], change, color='green' if change > 0 else 'red') +
            show_summary_card("Model Time", runtime, None, color='gray')
        )
        display(HTML(f"<div style='display:flex;'>{html_cards}</div>"))

        # --- Plotting ---
        plt.figure(figsize=(12, 6))
        plt.plot(df['Date'], df['Close'], label='Historical', color='black')
        plt.plot(future_dates, lstm, label='LSTM', marker='o', color='blue')
        plt.plot(future_dates, arima, label='ARIMA', marker='o', color='green')
        plt.plot(future_dates, prophet, label='Prophet', marker='o', color='orange')
        plt.plot(future_dates, ensemble, label='Ensemble', marker='X', color='red', linewidth=2)
        plt.axvline(x=last_date, color='gray', linestyle='--')
        plt.title(f'{ticker} Forecast for Next {days} Day{"s" if days > 1 else ""}')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # --- CSV Export ---
        if export_check.value:
            filename = f'{ticker}_forecast_{datetime.today().strftime("%Y%m%d")}.csv'
            results.to_csv(filename, index=False)
            print(f"✅ Forecast exported as: {filename}")

# --- Button Handlers ---
def on_run_clicked(b):
    ticker = custom_ticker.value.upper().strip() if custom_ticker.value else ticker_dropdown.value
    if not ticker:
        with output_area:
            clear_output()
            print("❌ Please enter a valid stock ticker.")
            return
    if ticker not in ticker_dropdown.options:
        ticker_dropdown.options = [*popular_tickers, ticker]
    run_forecast(ticker, start_date.value, forecast_days.value)

def on_clear_clicked(b):
    with output_area:
        clear_output()

run_btn.on_click(on_run_clicked)
clear_btn.on_click(on_clear_clicked)
