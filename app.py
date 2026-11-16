import io
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Apple Stock Price Prediction", page_icon="📈", layout="wide")
DATA_FILE = Path(__file__).parent / "P721 DATASET.csv"
LOOKBACK = 30
FORECAST_HORIZON = 30

@st.cache_data
def load_data(uploaded_bytes=None):
    if uploaded_bytes is not None:
        raw = pd.read_csv(io.BytesIO(uploaded_bytes))
    elif DATA_FILE.exists():
        raw = pd.read_csv(DATA_FILE)
    else:
        raise FileNotFoundError("P721 DATASET.csv is missing. Upload the CSV in the sidebar.")
    required = {"Date", "Close"}
    if not required.issubset(raw.columns):
        raise ValueError("CSV must contain at least Date and Close columns.")
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    raw["Close"] = pd.to_numeric(raw["Close"], errors="coerce")
    raw = raw.dropna(subset=["Date", "Close"]).sort_values("Date")
    raw = raw.drop_duplicates(subset="Date", keep="last").set_index("Date")
    if len(raw) < 100:
        raise ValueError("At least 100 valid daily observations are required.")
    return raw


def metric_row(actual, predicted, model_name):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mae = mean_absolute_error(actual, predicted)
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    mape = float(np.mean(np.abs((actual - predicted) / np.maximum(np.abs(actual), 1e-8))) * 100)
    return {"Model": model_name, "MAE": mae, "RMSE": rmse, "MAPE (%)": mape}


def make_sequences(values, lookback):
    X, y = [], []
    for i in range(lookback, len(values)):
        X.append(values[i-lookback:i, 0])
        y.append(values[i, 0])
    return np.asarray(X), np.asarray(y)


def train_neural_model(model_name, train_values, test_values, epochs):
    # Imported only when LSTM/GRU training is requested.
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping

    tf.keras.utils.set_random_seed(42)
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_values.reshape(-1, 1))
    combined_scaled = scaler.transform(np.concatenate([train_values, test_values]).reshape(-1, 1))
    X_train, y_train = make_sequences(train_scaled, LOOKBACK)
    test_start = len(train_values)
    X_test, _ = make_sequences(combined_scaled[test_start-LOOKBACK:], LOOKBACK)
    X_train = X_train.reshape((-1, LOOKBACK, 1))
    X_test = X_test.reshape((-1, LOOKBACK, 1))

    recurrent = LSTM if model_name == "LSTM" else GRU
    model = Sequential([
        recurrent(64, return_sequences=True, input_shape=(LOOKBACK, 1)),
        Dropout(0.2),
        recurrent(32),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    stop = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=epochs, batch_size=32,
              validation_split=0.1, callbacks=[stop], shuffle=False, verbose=0)
    scaled_pred = model.predict(X_test, verbose=0)
    pred = scaler.inverse_transform(scaled_pred).flatten()
    return pred


def run_experiment(close, epochs):
    split_idx = int(len(close) * 0.80)
    train = close.iloc[:split_idx]
    test = close.iloc[split_idx:]
    if len(test) == 0 or len(train) <= LOOKBACK + 20:
        raise ValueError("Not enough observations for the chronological 80/20 split.")

    predictions, metric_rows, errors = {}, [], {}
    progress = st.progress(0, text="Training ARIMA...")

    arima_fit = ARIMA(train, order=(5, 1, 0)).fit()
    arima_pred = np.asarray(arima_fit.forecast(steps=len(test)))
    predictions["ARIMA"] = pd.Series(arima_pred, index=test.index)
    metric_rows.append(metric_row(test, arima_pred, "ARIMA"))
    progress.progress(25, text="Training SARIMA...")

    sarima_fit = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 5),
                         enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    sarima_pred = np.asarray(sarima_fit.forecast(steps=len(test)))
    predictions["SARIMA"] = pd.Series(sarima_pred, index=test.index)
    metric_rows.append(metric_row(test, sarima_pred, "SARIMA"))
    progress.progress(50, text="Training LSTM...")

    for pct, name in [(75, "LSTM"), (100, "GRU")]:
        try:
            pred = train_neural_model(name, train.to_numpy(), test.to_numpy(), epochs)
            predictions[name] = pd.Series(pred, index=test.index)
            metric_rows.append(metric_row(test, pred, name))
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"
        progress.progress(pct, text=f"Completed {name} step...")

    progress.empty()
    metrics = pd.DataFrame(metric_rows).set_index("Model").sort_values("RMSE")
    best_name = metrics.index[0]
    future_dates = pd.bdate_range(start=close.index[-1] + pd.offsets.BDay(1), periods=FORECAST_HORIZON)

    if best_name == "ARIMA":
        future_values = np.asarray(ARIMA(close, order=(5, 1, 0)).fit().forecast(steps=FORECAST_HORIZON))
    elif best_name == "SARIMA":
        future_values = np.asarray(SARIMAX(close, order=(1, 1, 1), seasonal_order=(1, 1, 1, 5),
            enforce_stationarity=False, enforce_invertibility=False).fit(disp=False).forecast(steps=FORECAST_HORIZON))
    else:
        # Refit the selected recurrent model on all historical observations.
        import tensorflow as tf
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
        tf.keras.utils.set_random_seed(42)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(close.to_numpy().reshape(-1, 1))
        X_full, y_full = make_sequences(scaled, LOOKBACK)
        X_full = X_full.reshape((-1, LOOKBACK, 1))
        recurrent = LSTM if best_name == "LSTM" else GRU
        model = Sequential([recurrent(64, return_sequences=True, input_shape=(LOOKBACK, 1)),
            Dropout(0.2), recurrent(32), Dropout(0.2), Dense(1)])
        model.compile(optimizer="adam", loss="mean_squared_error")
        model.fit(X_full, y_full, epochs=epochs, batch_size=32, shuffle=False, verbose=0)
        window = scaled[-LOOKBACK:].copy()
        scaled_future = []
        for _ in range(FORECAST_HORIZON):
            nxt = model.predict(window.reshape(1, LOOKBACK, 1), verbose=0)[0, 0]
            scaled_future.append(nxt)
            window = np.vstack([window[1:], [[nxt]]])
        future_values = scaler.inverse_transform(np.asarray(scaled_future).reshape(-1, 1)).flatten()

    forecast = pd.DataFrame({"Date": future_dates, "Predicted_Close": future_values, "Selected_Model": best_name})
    test_predictions = pd.DataFrame({"Actual": test})
    for name, series in predictions.items():
        test_predictions[name] = series
    return metrics, predictions, test, forecast, test_predictions, errors

st.title("🍎 Apple Stock Price Prediction")
st.caption("Compare ARIMA, SARIMA, LSTM and GRU, evaluate forecasts, and project the next 30 trading-day observations.")
st.warning("Educational demonstration only. The supplied dataset covers 2012–2019; forecasts are not live 2026 market prices or investment advice.")

with st.sidebar:
    st.header("Project settings")
    uploaded = st.file_uploader("Upload a different Apple CSV (optional)", type=["csv"])
    epochs = st.slider("LSTM / GRU maximum epochs", min_value=5, max_value=50, value=30, step=5)
    st.caption("The app uses an 80/20 chronological split, 30-observation lookback, and the model parameters from the notebook.")

try:
    df = load_data(uploaded.getvalue() if uploaded else None)
    close = df["Close"].astype(float)
except Exception as exc:
    st.error(str(exc))
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Historical observations", f"{len(close):,}")
c2.metric("First date", close.index.min().strftime("%d %b %Y"))
c3.metric("Last date", close.index.max().strftime("%d %b %Y"))

st.subheader("Historical Apple closing price")
historical_chart = close.rename("Closing Price").to_frame()
st.line_chart(historical_chart)
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(close.index, close.values, linewidth=1.4)
ax.set_xlabel("Date"); ax.set_ylabel("Close price (USD)"); ax.grid(alpha=0.25)
st.pyplot(fig, use_container_width=True);
plt.close(fig)

st.subheader("Model training and evaluation")
st.write("Click below to train the four notebook models on this dataset. The first run may take several minutes, especially for LSTM and GRU.")
if st.button("Train and compare all four models", type="primary", use_container_width=True):
    try:
        with st.spinner("Fitting models and preparing the 30-step forecast..."):
            result = run_experiment(close, epochs)
        st.session_state["apple_results"] = result
    except Exception as exc:
        st.error(f"Training failed: {type(exc).__name__}: {exc}")

if "apple_results" in st.session_state:
    metrics, predictions, test, forecast, test_predictions, errors = st.session_state["apple_results"]
    st.subheader("Evaluation metrics (held-out test set)")
    st.dataframe(metrics.style.format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "MAPE (%)": "{:.2f}%"}), use_container_width=True)
    st.success(f"Lowest test RMSE: {metrics.index[0]}")
    if errors:
        for model, message in errors.items():
            st.warning(f"{model} could not be trained in this environment: {message}")

    st.subheader("Actual vs predicted closing prices")
    st.line_chart(test_predictions)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(test.index, test.values, label="Actual", color="black", linewidth=2)
    for name, pred in predictions.items():
        ax.plot(pred.index, pred.values, label=name, alpha=0.85)
    ax.set_xlabel("Date"); ax.set_ylabel("Close price (USD)"); ax.legend(); ax.grid(alpha=0.2)
    st.pyplot(fig, use_container_width=True);
    plt.close(fig)

    st.subheader("Next 30-day forecast")
    selected_model = forecast["Selected_Model"].iloc[0]
    first_forecast = float(forecast["Predicted_Close"].iloc[0])
    last_forecast = float(forecast["Predicted_Close"].iloc[-1])
    change_pct = ((last_forecast - first_forecast) / first_forecast * 100) if first_forecast else 0.0
    st.caption(f"Forecast generated using **{selected_model}**, selected because it had the lowest RMSE on the held-out test set.")
    f1, f2, f3 = st.columns(3)
    f1.metric("First forecast close", f"${first_forecast:,.2f}", forecast["Date"].iloc[0].strftime("%d %b %Y"))
    f2.metric("30th forecast close", f"${last_forecast:,.2f}", forecast["Date"].iloc[-1].strftime("%d %b %Y"))
    f3.metric("Change across forecast", f"{change_pct:+.2f}%")
    st.info("These are 30 future weekdays generated from the last date in the uploaded dataset. Weekends are excluded; exchange holidays are not adjusted. They are model estimates, not live quotes or guaranteed prices.")
   # Prepare data for the forecast chart
st.subheader("Forecast chart")

recent_history = close.tail(120).copy()
recent_history.index = pd.to_datetime(recent_history.index)
recent_history = recent_history.rename("Historical Close")

forecast_chart = forecast.copy()
forecast_chart["Date"] = pd.to_datetime(forecast_chart["Date"])

future_chart = forecast_chart.set_index("Date")["Predicted_Close"]
future_chart = future_chart.rename("Forecast Close")

chart_data = pd.concat([recent_history, future_chart], axis=1)

st.line_chart(chart_data) 
    st.dataframe(forecast, use_container_width=True)

    st.download_button("Download 30-day forecast CSV", forecast.to_csv(index=False).encode("utf-8"),
                       file_name="apple_next_30_weekdays_forecast.csv", mime="text/csv")
    st.download_button("Download model metrics CSV", metrics.to_csv().encode("utf-8"),
                       file_name="apple_model_metrics.csv", mime="text/csv")
    st.download_button("Download test predictions CSV", test_predictions.to_csv(index_label="Date").encode("utf-8"),
                       file_name="apple_model_test_predictions.csv", mime="text/csv")
else:
    st.info("The historical chart is ready. Start model training to display evaluation metrics, model comparisons, and the 30-step forecast.")

st.divider()
st.caption("Methodology follows the project notebook: Close is the target; 80/20 chronological split; ARIMA(5,1,0); SARIMA(1,1,1) with seasonal (1,1,1,5); LSTM/GRU with 30-step lookback; MAE, RMSE and MAPE evaluation.")
