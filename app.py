from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
import os
import sqlite3
from datetime import datetime
from pathlib import Path
import pandas as pd
from ml.model import ExpenseForecaster

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "expenses.db"
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-only-change-me")


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
            amount REAL NOT NULL CHECK(amount > 0),
            notes TEXT DEFAULT ''
        )""")
        conn.commit()


def rows_to_df(rows):
    if not rows:
        return pd.DataFrame(columns=["date", "category", "amount", "notes"])
    df = pd.DataFrame([dict(r) for r in rows])
    df["date"] = pd.to_datetime(df["date"])
    df["amount"] = pd.to_numeric(df["amount"])
    return df


def get_forecaster():
    return ExpenseForecaster(DB_PATH)


@app.route("/")
def dashboard():
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM expenses ORDER BY date DESC, id DESC").fetchall()
    df = rows_to_df(rows)
    total = float(df["amount"].sum()) if not df.empty else 0.0
    today = pd.Timestamp.today().normalize()
    month_total = float(df.loc[df["date"] >= today.replace(day=1), "amount"].sum()) if not df.empty else 0.0
    avg_daily = float(df.groupby("date")["amount"].sum().mean()) if not df.empty else 0.0
    categories = df.groupby("category")["amount"].sum().sort_values(ascending=False).to_dict() if not df.empty else {}
    daily = df.groupby("date")["amount"].sum().sort_index().tail(30) if not df.empty else pd.Series(dtype=float)
    recent = [dict(r) for r in rows[:8]]
    return render_template("dashboard.html", total=total, month_total=month_total, avg_daily=avg_daily,
                           categories=categories, daily=daily, recent=recent, expense_count=len(df))


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
        except (TypeError, ValueError):
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


@app.post("/expenses/edit/<int:expense_id>")
def edit_expense(expense_id):
    date = request.form.get("date", "").strip()
    category = request.form.get("category", "").strip()
    notes = request.form.get("notes", "").strip()
    try:
        amount = float(request.form.get("amount", "0"))
        datetime.strptime(date, "%Y-%m-%d")
        if amount <= 0 or not category:
            raise ValueError
    except (TypeError, ValueError):
        flash("Invalid expense details.", "error")
        return redirect(url_for("expenses"))
    with get_db() as conn:
        updated = conn.execute("UPDATE expenses SET date=?, category=?, amount=?, notes=? WHERE id=?",
                               (date, category, amount, notes, expense_id)).rowcount
        conn.commit()
    flash("Expense updated." if updated else "Expense not found.", "success" if updated else "error")
    return redirect(url_for("expenses"))


@app.post("/expenses/delete/<int:expense_id>")
def delete_expense(expense_id):
    with get_db() as conn:
        deleted = conn.execute("DELETE FROM expenses WHERE id = ?", (expense_id,)).rowcount
        conn.commit()
    flash("Expense deleted." if deleted else "Expense not found.", "success" if deleted else "error")
    return redirect(url_for("expenses"))


@app.route("/analytics")
def analytics():
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM expenses ORDER BY date").fetchall()
    df = rows_to_df(rows)
    category_data = df.groupby("category")["amount"].sum().sort_values(ascending=False) if not df.empty else pd.Series(dtype=float)
    monthly = df.assign(month=df["date"].dt.to_period("M")).groupby("month")["amount"].sum() if not df.empty else pd.Series(dtype=float)
    return render_template("analytics.html", category_data=category_data.to_dict(), monthly={str(k): float(v) for k, v in monthly.items()})


@app.route("/predict")
def predict():
    try:
        days = int(request.args.get("days", 7))
    except ValueError:
        days = 7
    days = max(1, min(days, 30))
    return render_template("predictions.html", result=get_forecaster().predict(days), days=days)


@app.get("/api/predict")
def predict_api():
    try:
        days = int(request.args.get("days", 7))
    except ValueError:
        return jsonify({"error": "days must be an integer"}), 400
    if not 1 <= days <= 30:
        return jsonify({"error": "days must be between 1 and 30"}), 400
    return jsonify(get_forecaster().predict(days))


@app.get("/api/dashboard")
def dashboard_api():
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM expenses ORDER BY date").fetchall()
    df = rows_to_df(rows)
    daily = df.groupby("date")["amount"].sum() if not df.empty else pd.Series(dtype=float)
    return jsonify({"total": float(df["amount"].sum()) if not df.empty else 0,
                    "transactions": len(df),
                    "daily": [{"date": d.strftime("%Y-%m-%d"), "amount": float(v)} for d, v in daily.tail(30).items()]})


@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "expense-tracker"})


@app.context_processor
def inject_now():
    return {"current_year": datetime.now().year}


init_db()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=os.environ.get("FLASK_DEBUG") == "1")
