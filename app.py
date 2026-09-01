from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

from ml.model import ExpenseForecaster

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "expenses.db"

app = Flask(__name__)
app.secret_key = "expense-tracker-dev-key"
forecaster = ExpenseForecaster(DB_PATH)


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS expenses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            category TEXT NOT NULL,
            amount REAL NOT NULL CHECK(amount >= 0),
            notes TEXT DEFAULT ''
        )""")
        conn.commit()


def rows_to_df(rows):
    if not rows:
        return pd.DataFrame(columns=["date", "category", "amount", "notes"])
    df = pd.DataFrame([dict(r) for r in rows])
    df["date"] = pd.to_datetime(df["date"])
    return df


@app.route("/")
def dashboard():
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM expenses ORDER BY date DESC, id DESC").fetchall()
    df = rows_to_df(rows)
    total = float(df["amount"].sum()) if not df.empty else 0.0
    today = pd.Timestamp.today().normalize()
    month_start = today.replace(day=1)
    month_total = float(df.loc[df["date"] >= month_start, "amount"].sum()) if not df.empty else 0.0
    avg_daily = float(df.groupby("date")["amount"].sum().mean()) if not df.empty else 0.0
    categories = df.groupby("category")["amount"].sum().sort_values(ascending=False).to_dict() if not df.empty else {}
    daily = df.groupby("date")["amount"].sum().sort_index().tail(30) if not df.empty else pd.Series(dtype=float)
    recent = [dict(r) for r in rows[:8]]
    return render_template("dashboard.html", total=total, month_total=month_total,
                           avg_daily=avg_daily, categories=categories, daily=daily,
                           recent=recent, expense_count=len(df))


@app.route("/expenses", methods=["GET", "POST"])
def expenses():
    if request.method == "POST":
        date = request.form.get("date", "").strip()
        category = request.form.get("category", "").strip()
        notes = request.form.get("notes", "").strip()
        try:
            amount = float(request.form.get("amount", "0"))
            datetime.strptime(date, "%Y-%m-%d")
            if amount <= 0 or not category:
                raise ValueError
        except ValueError:
            flash("Enter a valid date, category and positive amount.", "error")
            return redirect(url_for("expenses"))
        with get_db() as conn:
            conn.execute("INSERT INTO expenses(date, category, amount, notes) VALUES (?, ?, ?, ?)",
                         (date, category, amount, notes))
            conn.commit()
        flash("Expense added successfully.", "success")
        return redirect(url_for("expenses"))

    with get_db() as conn:
        rows = conn.execute("SELECT * FROM expenses ORDER BY date DESC, id DESC").fetchall()
    return render_template("expenses.html", expenses=rows, today=datetime.now().strftime("%Y-%m-%d"))


@app.post("/expenses/delete/<int:expense_id>")
def delete_expense(expense_id):
    with get_db() as conn:
        conn.execute("DELETE FROM expenses WHERE id = ?", (expense_id,))
        conn.commit()
    flash("Expense deleted.", "success")
    return redirect(url_for("expenses"))


@app.route("/analytics")
def analytics():
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM expenses ORDER BY date").fetchall()
    df = rows_to_df(rows)
    category_data = df.groupby("category")["amount"].sum().sort_values(ascending=False) if not df.empty else pd.Series(dtype=float)
    monthly = df.assign(month=df["date"].dt.to_period("M")).groupby("month")["amount"].sum() if not df.empty else pd.Series(dtype=float)
    return render_template("analytics.html", category_data=category_data.to_dict(), monthly=monthly.astype(float).to_dict())


@app.route("/predict")
def predict():
    days = max(1, min(int(request.args.get("days", 7)), 30))
    result = forecaster.predict(days)
    return render_template("predictions.html", result=result, days=days)


@app.get("/api/dashboard")
def dashboard_api():
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM expenses ORDER BY date").fetchall()
    df = rows_to_df(rows)
    daily = df.groupby("date")["amount"].sum() if not df.empty else pd.Series(dtype=float)
    return jsonify({
        "total": float(df["amount"].sum()) if not df.empty else 0,
        "transactions": len(df),
        "daily": [{"date": d.strftime("%Y-%m-%d"), "amount": float(v)} for d, v in daily.tail(30).items()]
    })


@app.context_processor
def inject_now():
    return {"current_year": datetime.now().year}


init_db()

if __name__ == "__main__":
    app.run(debug=True)
