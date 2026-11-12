import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

st.set_page_config(
    page_title="Apple Stock Price Prediction",
    page_icon="📈",
    layout="wide",
)

st.title("🍎 Apple Stock Price Prediction")
st.caption("Time-series analysis and model evaluation using ARIMA, SARIMA, LSTM and GRU.")

with st.sidebar:
    st.header("Project controls")
    uploaded_file = st.file_uploader(
        "Upload historical stock CSV",
        type=["csv"],
        help="CSV must contain Date and Close columns. Open, High, Low and Volume are recommended for EDA and deep learning."
    )
    st.divider()
    st.subheader("Model settings")
    test_fraction = st.slider("Test data proportion", 0.10, 0.40, 0.20, 0.05)
    lookback = st.slider("LSTM / GRU look-back window", 5, 60, 10, 1)
    epochs = st.slider("Deep-learning epochs", 5, 50, 15, 5)
    run_arima = st.checkbox("Run ARIMA", value=True)
    run_sarima = st.checkbox("Run SARIMA", value=True)
    run_lstm = st.checkbox("Run LSTM", value=True)
    run_gru = st.checkbox("Run GRU", value=True)
    run_button = st.button("Train and evaluate models", type="primary", use_container_width=True)

st.info(
    "Start by uploading the original **P721 DATASET.csv**. "
    "The app prepares date-based and volume features from the uploaded data."
)

@st.cache_data(show_spinner=False)
def load_and_prepare(file_bytes):
    from io import BytesIO
    data = pd.read_csv(BytesIO(file_bytes))
    data.columns = [str(c).strip() for c in data.columns]

    required = {"Date", "Close"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(sorted(missing))}")

    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
    data = data.dropna(subset=["Date", "Close"]).drop_duplicates(subset=["Date"])
    data = data.sort_values("Date").set_index("Date")

    for col in ["Open", "High", "Low", "Volume", "Adj Close"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    if "Volume" not in data.columns:
        data["Volume"] = 0.0
    data["Volume"] = data["Volume"].fillna(0).clip(lower=0)
    data["Volume_Log"] = np.log1p(data["Volume"])
    data["MA_7"] = data["Close"].rolling(7).mean()
    data["MA_30"] = data["Close"].rolling(30).mean()
    data["Daily_Return"] = data["Close"].pct_change() * 100
    data["Year"] = data.index.year
    data["Month"] = data.index.month
    data["Day"] = data.index.day
    data["Day_of_Week"] = data.index.dayofweek
    data["Quarter"] = data.index.quarter
    if {"High", "Low"}.issubset(data.columns):
        data["Price_Range"] = data["High"] - data["Low"]
    if {"Open"}.issubset(data.columns):
        data["Price_Change"] = data["Close"] - data["Open"]
    data = data.replace([np.inf, -np.inf], np.nan)
    return data

def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) == 0:
        raise ValueError("No valid aligned predictions are available.")
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    nonzero = np.abs(y_true) > 1e-12
    mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100) if nonzero.any() else np.nan
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan
    if len(y_true) > 1:
        dir_acc = float(np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))) * 100)
    else:
        dir_acc = np.nan
    return {"RMSE": rmse, "MAE": mae, "MAPE (%)": mape, "R²": r2, "Directional Accuracy (%)": dir_acc}

def fit_deep_model(frame, split_idx, lookback, epochs, kind):
    try:
        import tensorflow as tf
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
    except Exception as exc:
        raise RuntimeError(
            "TensorFlow could not be imported in this environment. "
            "Use the supplied requirements.txt and Python 3.11 on Streamlit Community Cloud. "
            f"Details: {exc}"
        )

    tf.keras.utils.set_random_seed(42)
    feature_cols = ["Close", "Volume_Log", "MA_7", "MA_30"]
    usable = frame[feature_cols].copy()
    # Train-only scaler fit: avoid leaking test-period information.
    scaler = MinMaxScaler()
    scaler.fit(usable.iloc[:split_idx])
    scaled = scaler.transform(usable)

    X_train, y_train, X_test, y_test = [], [], [], []
    test_dates = []
    for target_idx in range(lookback, len(scaled)):
        seq = scaled[target_idx-lookback:target_idx, :]
        if target_idx < split_idx:
            X_train.append(seq)
            y_train.append(scaled[target_idx, 0])
        else:
            X_test.append(seq)
            y_test.append(scaled[target_idx, 0])
            test_dates.append(frame.index[target_idx])

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("Not enough rows for the selected look-back window and train/test split.")

    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.float32)

    layer = LSTM if kind == "LSTM" else GRU
    model = Sequential([
        layer(50, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_split=0.1, verbose=0)

    pred_scaled = model.predict(X_test, verbose=0).reshape(-1)
    # Inverse transform only the Close column.
    close_min = scaler.data_min_[0]
    close_scale = scaler.data_range_[0]
    pred = pred_scaled * close_scale + close_min
    actual = y_test * close_scale + close_min
    pred_series = pd.Series(pred, index=pd.DatetimeIndex(test_dates), name=kind)
    actual_series = pd.Series(actual, index=pd.DatetimeIndex(test_dates), name="Actual")
    return pred_series, actual_series

if uploaded_file is None:
    st.subheader("About this application")
    st.markdown(
        """
        This Streamlit application is based on the Apple Stock Price Prediction project.

        **Workflow**
        1. Upload historical Apple stock data.
        2. Review the dataset and exploratory charts.
        3. Select forecasting models.
        4. Train models and compare test-period predictions.

        **Required CSV columns:** `Date` and `Close`.  
        **Recommended columns:** `Open`, `High`, `Low`, `Volume`, and `Adj Close`.

        The app evaluates historical test-period predictions. It does not present those
        values as guaranteed future stock prices or investment advice.
        """
    )
    st.stop()

try:
    df = load_and_prepare(uploaded_file.getvalue())
except Exception as exc:
    st.error(f"Could not read the uploaded CSV: {exc}")
    st.stop()

if len(df) < 80:
    st.error("Please upload a dataset with at least 80 valid dated observations.")
    st.stop()

# Overview
st.subheader("1. Dataset overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Observations", f"{len(df):,}")
c2.metric("First date", df.index.min().strftime("%Y-%m-%d"))
c3.metric("Last date", df.index.max().strftime("%Y-%m-%d"))
c4.metric("Latest close", f"${df['Close'].iloc[-1]:,.2f}")

with st.expander("Preview uploaded data"):
    st.dataframe(df.tail(10), use_container_width=True)

# EDA
st.subheader("2. Exploratory data analysis")
tab1, tab2, tab3 = st.tabs(["Closing price", "Volume", "Statistics"])
with tab1:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df.index, df["Close"], linewidth=1.4)
    ax.set_title("Apple historical closing price")
    ax.set_xlabel("Date")
    ax.set_ylabel("Closing price")
    ax.grid(alpha=0.25)
    st.pyplot(fig)
    plt.close(fig)
with tab2:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df.index, df["Volume"], linewidth=1)
    ax.set_title("Trading volume")
    ax.set_xlabel("Date")
    ax.set_ylabel("Volume")
    ax.grid(alpha=0.25)
    st.pyplot(fig)
    plt.close(fig)
with tab3:
    st.dataframe(df[["Close", "Volume", "Volume_Log", "MA_7", "MA_30"]].describe().T, use_container_width=True)
    st.write("Missing values by column")
    st.dataframe(df.isna().sum().rename("Missing values").to_frame(), use_container_width=True)

# Build evaluation on rows with engineered moving averages
model_df = df.dropna(subset=["MA_7", "MA_30", "Volume_Log"]).copy()
if len(model_df) < 80:
    st.error("Not enough observations remain after feature preparation.")
    st.stop()

split_idx = int(len(model_df) * (1 - test_fraction))
train = model_df.iloc[:split_idx]
test = model_df.iloc[split_idx:]
st.subheader("3. Chronological train/test split")
a, b, c = st.columns(3)
a.metric("Training rows", f"{len(train):,}")
b.metric("Testing rows", f"{len(test):,}")
c.metric("Training proportion", f"{(1-test_fraction)*100:.0f}%")

if run_button:
    results = {}
    prediction_series = {}
    progress = st.progress(0)
    status = st.empty()
    tasks = int(run_arima) + int(run_sarima) + int(run_lstm) + int(run_gru)
    if tasks == 0:
        st.warning("Select at least one model in the sidebar.")
        st.stop()
    done = 0

    # Baseline benchmark always included.
    baseline = test["Close"].shift(1)
    baseline.iloc[0] = train["Close"].iloc[-1]
    prediction_series["Naive Baseline"] = baseline
    results["Naive Baseline"] = metrics(test["Close"], baseline)

    if run_arima:
        status.info("Fitting ARIMA. This may take a little while...")
        try:
            from itertools import product
            best_order, best_aic = None, np.inf
            train_close = train["Close"].asfreq("B").ffill()
            for order in product(range(3), [1], range(3)):
                try:
                    fit = ARIMA(train_close, order=order).fit()
                    if fit.aic < best_aic:
                        best_aic, best_order = fit.aic, order
                except Exception:
                    continue
            if best_order is None:
                raise RuntimeError("ARIMA parameter search did not find a valid model.")
            fit = ARIMA(train_close, order=best_order).fit()
            pred = fit.forecast(steps=len(test))
            pred.index = test.index
            prediction_series[f"ARIMA {best_order}"] = pred
            results[f"ARIMA {best_order}"] = metrics(test["Close"], pred)
            st.session_state["arima_order"] = best_order
        except Exception as exc:
            st.warning(f"ARIMA could not be fitted: {exc}")
        done += 1
        progress.progress(done / tasks)

    if run_sarima:
        status.info("Fitting SARIMA...")
        try:
            order = st.session_state.get("arima_order", (1, 1, 1))
            train_close = train["Close"].asfreq("B").ffill()
            fit = SARIMAX(
                train_close, order=order, seasonal_order=(1, 1, 1, 5),
                enforce_stationarity=False, enforce_invertibility=False
            ).fit(disp=False)
            pred = fit.forecast(steps=len(test))
            pred.index = test.index
            prediction_series["SARIMA"] = pred
            results["SARIMA"] = metrics(test["Close"], pred)
        except Exception as exc:
            st.warning(f"SARIMA could not be fitted: {exc}")
        done += 1
        progress.progress(done / tasks)

    if run_lstm:
        status.info("Training LSTM neural network...")
        try:
            pred, actual_dl = fit_deep_model(model_df, split_idx, lookback, epochs, "LSTM")
            prediction_series["LSTM"] = pred
            results["LSTM"] = metrics(actual_dl, pred)
        except Exception as exc:
            st.warning(f"LSTM could not be trained: {exc}")
        done += 1
        progress.progress(done / tasks)

    if run_gru:
        status.info("Training GRU neural network...")
        try:
            pred, actual_dl = fit_deep_model(model_df, split_idx, lookback, epochs, "GRU")
            prediction_series["GRU"] = pred
            results["GRU"] = metrics(actual_dl, pred)
        except Exception as exc:
            st.warning(f"GRU could not be trained: {exc}")
        done += 1
        progress.progress(done / tasks)

    status.empty()
    progress.empty()

    if results:
        summary = pd.DataFrame(results).T
        st.session_state["model_summary"] = summary
        st.session_state["model_predictions"] = prediction_series
        st.session_state["test_actual"] = test["Close"]
        st.session_state["test_dates"] = test.index
        st.success("Model evaluation completed. Review the results below.")
    else:
        st.error("No model completed successfully. Check the messages above.")

if "model_summary" in st.session_state:
    st.subheader("4. Model performance comparison")
    summary = st.session_state["model_summary"]
    st.dataframe(summary.style.format({
        "RMSE": "{:.4f}", "MAE": "{:.4f}", "MAPE (%)": "{:.3f}",
        "R²": "{:.4f}", "Directional Accuracy (%)": "{:.2f}"
    }), use_container_width=True)

    st.subheader("5. Actual vs predicted closing prices")
    fig, ax = plt.subplots(figsize=(13, 5))
    actual = st.session_state["test_actual"]
    ax.plot(actual.index, actual.values, label="Actual", color="black", linewidth=2)
    for name, series in st.session_state["model_predictions"].items():
        ax.plot(series.index, series.values, label=name, alpha=0.85)
    ax.set_xlabel("Date")
    ax.set_ylabel("Closing price")
    ax.set_title("Actual vs predicted prices on the test period")
    ax.legend()
    ax.grid(alpha=0.25)
    st.pyplot(fig)
    plt.close(fig)

    csv = summary.reset_index(names="Model").to_csv(index=False).encode("utf-8")
    st.download_button("Download model comparison (CSV)", csv, "apple_model_comparison.csv", "text/csv")

st.divider()
st.caption(
    "Educational project demonstration only. Historical model performance is not a guarantee "
    "of future returns and this application is not investment advice."
)
