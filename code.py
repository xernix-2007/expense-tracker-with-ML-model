import csv
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import numpy as np

csv_file = "expense.csv"
try:
    with open(csv_file, "x", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "category", "amount", "notes"])
except FileExistsError:
    pass


def add_expense():
    date = input("Date (dd/mm/yyyy): ").strip()
    category = input("Category: ").strip()

    while True:
        amount = input("Amount: ").strip()
        if amount.replace(".", "", 1).isdigit():
            break
        else:
            print("Enter a valid number (like 100 or 99.50)")

    notes = input("Notes: ").strip()

    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([date, category, amount, notes])

    print("Expense added successfully!")


def load_data():
    """Load CSV into a DataFrame and clean types."""
    df = pd.read_csv(csv_file)

    if df.empty:
        return df
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
    df = df.dropna(subset=["date", "amount"])

    return df
def view_expenses():
    df = load_data()

    if df.empty:
        print("No expenses recorded yet!")
        return

    print("\nAll expenses:\n")
    print(df.to_string(index=False))

    total = df["amount"].sum()
    print(f"\nTotal expenses: {total:.2f}")


def predict_future_expenses():
    df = load_data()

    if df.empty:
        print("No data available to train the model.")
        return
    daily = df.groupby("date")["amount"].sum().reset_index()

    if len(daily) < 2:
        print("Need at least 2 different days of data to predict.")
        return
    daily = daily.sort_values("date").reset_index(drop=True)
    daily["day_num"] = range(len(daily))

    X = daily[["day_num"]].values 
    y = daily["amount"].values    

    model = LinearRegression()
    model.fit(X, y)
    while True:
        days_str = input("How many future days to predict? ").strip()
        if days_str.isdigit() and int(days_str) > 0:
            days_ahead = int(days_str)
            break
        else:
            print("Enter a positive whole number (like 7 or 30).")

    last_day_num = daily["day_num"].iloc[-1]
    last_date = daily["date"].iloc[-1]
    future_day_nums = np.array(
        [last_day_num + i for i in range(1, days_ahead + 1)]
    ).reshape(-1, 1)

    future_dates = [last_date + timedelta(days=i) for i in range(1, days_ahead + 1)]

    predicted_amounts = model.predict(future_day_nums)

    print("\nPredicted future expenses:\n")
    for d, amt in zip(future_dates, predicted_amounts):
        print(d.strftime("%d/%m/%Y"), "->", f"{amt:.2f}")


def main():
    while True:
        print("\n--- Expense Tracker ---")
        print("1. Add expense")
        print("2. View expenses")
        print("3. Predict future expenses")
        print("4. Exit")

        choice = input("Choose an option: ").strip()

        if choice == "1":
            add_expense()
        elif choice == "2":
            view_expenses()
        elif choice == "3":
            predict_future_expenses()
        elif choice == "4":
            print("Goodbye!")
            break
        else:
            print("Invalid option. Try again.")


if __name__ == "__main__":
    main()
