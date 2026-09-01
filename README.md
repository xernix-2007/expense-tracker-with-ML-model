# 💳 SpendWise — Expense Tracker + ML Forecasting

SpendWise is a full-stack personal expense tracker built with **Flask, HTML/CSS, Chart.js, SQLite and scikit-learn**. It upgrades the original CLI tracker into a usable web application with analytics and machine-learning forecasting.

## ✨ Features
- Responsive dashboard with KPI cards
- Add, edit and delete expenses
- SQLite persistent storage
- Category and monthly analytics
- Spending trend and category charts
- ML-powered daily expense forecasting
- Chronological validation metrics: MAE, RMSE and R² when calculable
- JSON dashboard and prediction APIs
- Health endpoint for deployment checks
- Reproducible demo-data seeder
- Production WSGI start command

## 🧠 ML dataset
There is **no external Kaggle dataset pretending to represent the user**. The forecasting model is trained on the expense history stored by the application.

Each transaction contains:
`date`, `category`, `amount`, `notes`

For the forecasting task, transactions are aggregated by date:

```text
Transactions → Daily total spending → Time feature → ML model → Future daily forecast
```

The repository includes `seed_demo.py` to generate **60 days of clearly synthetic demo spending data**. This is only for demonstrating the dashboard immediately; it is not claimed to be real user data.

The model requires expenses across at least 3 different days. As the application is used, real tracker history can replace the demo data.

## 🤖 Model
The current model is **Polynomial Regression (degree 2)** over elapsed day number. An 80/20 chronological validation split is used when enough history exists. The service reports MAE/RMSE/R² and then retrains the final model on all available history for forecasting.

This is a portfolio/learning forecasting system, not financial advice. With limited history, predictions can be poor; more history and stronger time-series features are required for serious forecasting.

## 🏗️ Architecture

```text
Browser
  │
  ├── HTML/CSS + Chart.js
  │
  ▼
Flask application
  │
  ├── SQLite database
  │      └── expense history
  │
  └── ML service
         ├── daily aggregation
         ├── chronological validation
         ├── Polynomial Regression
         └── forecast + metrics
```

## 🚀 Run locally

```bash
python -m venv .venv

# Windows
.venv\\Scripts\\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
python seed_demo.py   # optional: creates 60 days of demo data
python app.py
```

Open `http://127.0.0.1:5000`.

To start production-style with Gunicorn:

```bash
gunicorn app:app
```

## 📁 Important files
- `app.py` — Flask application, routes and database access
- `ml/model.py` — training, validation and prediction service
- `seed_demo.py` — reproducible synthetic demo dataset
- `templates/` — dashboard, expenses, analytics and ML pages
- `static/css/style.css` — responsive UI
- `requirements.txt` — Python dependencies
- `Procfile` — WSGI web start command

## 🔌 API
- `GET /health` — service health
- `GET /api/dashboard` — dashboard totals and daily history
- `GET /api/predict?days=7` — ML forecast for 1–30 days

## ⚠️ Limitations
The current model uses a simple trend feature and should not be described as an advanced financial forecasting model. Future improvements could include richer calendar/category features, stronger time-series baselines, authentication, budgets, CSV import/export, tests, CI and Docker deployment.
