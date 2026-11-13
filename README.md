# Apple Stock Price Prediction — Streamlit App

This Streamlit application converts the project notebook's four-model workflow into an interactive web app.

## Files
- `app.py` — Streamlit application
- `P721 DATASET.csv` — supplied historical Apple dataset (2012–2019)
- `requirements.txt` — deployment dependencies
- `.python-version` — requests Python 3.11 on supported hosts

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Upload all four files to the root of a GitHub repository.
2. In Streamlit Community Cloud, choose **Create app**.
3. Select the repository, branch `main`, and main file path `app.py`.
4. Deploy. The app trains the models only when the user clicks **Train and compare all four models**.

## Methodology
- Target: `Close`
- Chronological 80/20 train/test split
- ARIMA order: `(5, 1, 0)`
- SARIMA order: `(1, 1, 1)`; seasonal order: `(1, 1, 1, 5)`
- LSTM and GRU: 30-observation lookback, MinMax scaling fitted on training data, 64/32 recurrent units and dropout
- Metrics: MAE, RMSE, MAPE
- Forecast: 30 future business-day observations, using the model with the lowest test RMSE among successfully trained models

## Important limitation
The dataset ends in 2019. The forecasts are a historical-data demonstration, not live/current stock prices or investment advice. Future business dates are approximated with weekdays and do not exclude exchange holidays.
