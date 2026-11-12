# Apple Stock Price Prediction — Streamlit App

A Streamlit dashboard for the Apple Stock Price Prediction project. It accepts a historical stock CSV, performs basic preprocessing and EDA, and evaluates the Naive Baseline, ARIMA, SARIMA, LSTM and GRU models.

## 1. Dataset

Upload `P721 DATASET.csv` from the app sidebar.

Required columns:
- `Date`
- `Close`

Recommended columns:
- `Open`
- `High`
- `Low`
- `Volume`
- `Adj Close`

## 2. Run locally (Windows PowerShell)

Open PowerShell in this folder and run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks environment activation, use Command Prompt instead:

```cmd
.venv\Scripts\activate.bat
pip install -r requirements.txt
streamlit run app.py
```

Use Python 3.11 for the pinned TensorFlow dependency.

## 3. Deploy on Streamlit Community Cloud

1. Create a GitHub repository, for example `apple-stock-price-prediction`.
2. Upload `app.py`, `requirements.txt`, and `README.md` to the repository root.
3. Push/commit the files.
4. Open https://share.streamlit.io/ and sign in with GitHub.
5. Select **Create app**.
6. Choose your repository, branch, and `app.py` as the main file.
7. In Advanced settings, select Python 3.11 if the option is available.
8. Click **Deploy**.
9. Open the deployed app and upload `P721 DATASET.csv` through the sidebar.

## Notes

- The app uses a chronological train/test split to preserve time order.
- The MinMaxScaler for LSTM/GRU is fitted on training observations only.
- The app reports test-period evaluation, not guaranteed future stock prices.
- TensorFlow training can take several minutes on a CPU. You can lower epochs in the sidebar.
- This is an educational project, not investment advice.
