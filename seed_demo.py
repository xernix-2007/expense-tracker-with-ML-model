import sqlite3
from datetime import date, timedelta
from pathlib import Path
import random

DB_PATH = Path(__file__).resolve().parent / "expenses.db"
random.seed(42)
categories = [("Food", 0.28), ("Transport", 0.15), ("Shopping", 0.16), ("Bills", 0.18),
              ("Entertainment", 0.08), ("Education", 0.07), ("Health", 0.05), ("Other", 0.03)]
weights = [w for _, w in categories]
names = [n for n, _ in categories]
notes = {"Food": "Lunch / groceries", "Transport": "Commute", "Shopping": "Personal purchase",
         "Bills": "Monthly bill", "Entertainment": "Weekend", "Education": "Course / books",
         "Health": "Pharmacy", "Other": "Miscellaneous"}

with sqlite3.connect(DB_PATH) as conn:
    conn.execute("CREATE TABLE IF NOT EXISTS expenses (id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT NOT NULL, category TEXT NOT NULL, amount REAL NOT NULL CHECK(amount > 0), notes TEXT DEFAULT '')")
    conn.execute("DELETE FROM expenses")
    start = date.today() - timedelta(days=59)
    for i in range(60):
        d = start + timedelta(days=i)
        count = random.randint(1, 4)
        for _ in range(count):
            category = random.choices(names, weights=weights, k=1)[0]
            base = {"Food": 180, "Transport": 100, "Shopping": 450, "Bills": 900,
                    "Entertainment": 300, "Education": 250, "Health": 350, "Other": 150}[category]
            amount = max(20, random.gauss(base, base * 0.35))
            conn.execute("INSERT INTO expenses(date, category, amount, notes) VALUES (?, ?, ?, ?)",
                         (d.isoformat(), category, round(amount, 2), notes[category]))
    conn.commit()
print("Seeded 60 days of demo expense data into expenses.db")
