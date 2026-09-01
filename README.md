# 💳 SpendWise — Expense Tracker + ML Forecasting

A full-stack personal expense tracker built with Flask, HTML/CSS, SQLite and machine learning. The original project was a CLI that stored expenses in CSV and used basic Linear Regression; it is now upgraded into a usable web application with analytics and an ML forecasting page.

## Features
- Responsive dashboard with KPI cards and charts
- Add and delete expenses
- SQLite persistent storage
- Category and monthly analytics
- ML-powered daily expense forecasting
- Validation metrics (MAE, RMSE and R² when available)
- JSON dashboard endpoint
- Clean separation of web, database and ML logic

## Dataset used for ML
**The production model is trained on the user's recorded tracker data, not a random Kaggle dataset.** Each expense entered in the application is stored in SQLite. For forecasting, transactions are aggregated by date into daily spending totals.

This is deliberately a **personal time-series forecasting problem**. A generic public dataset would not accurately represent an individual's spending behavior. The repository starts without fabricated personal spending history, so the app does not pretend that invented data represents the user.

The ML forecast becomes available after expenses exist on at least 3 different days. As the app is used, the training dataset grows naturally from real tracker history.

## Model
The forecasting service uses Polynomial Regression (degree 2) over elapsed day number. It performs an 80/20 chronological validation split when enough observations exist, reports MAE/RMSE/R², then retrains on all available history for production forecasts. Predictions are clipped at zero because negative spending is not meaningful.

> This is a learning/project forecasting system, not financial advice.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

## Architecture

```text
Browser (HTML/CSS + Chart.js)
          ↓
       Flask app
       ↙       ↘
   SQLite       ML service
                   ↓
          Daily spending history
                   ↓
          Polynomial Regression
                   ↓
             Forecast + metrics
```

## Next improvements
- User authentication
- Edit transactions
- Budget limits and alerts
- Better time-series models after enough history exists
- CSV import/export
- Docker deployment
- Automated tests and CI
