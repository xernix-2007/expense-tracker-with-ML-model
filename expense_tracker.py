import csv
import pandas as pd
from datetime import datetime, timedelta # Ensure these are used when needed
from sklearn.linear_model import LinearRegression
import numpy as np

csv_file = "expense.csv"
# Create file with headers if it doesn't exist
try:
    with open(csv_file, "x", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "category", "amount", "notes"])
except FileExistsError:
    pass


def add_expense():
    # Input validation for date format
    while True:
        date_str = input("Date (dd/mm/yyyy): ").strip()
        try:
            # Validate format against the standard used in load_data
            datetime.strptime(date_str, "%d/%m/%Y")
            date = date_str
            break
        except ValueError:
            print("Invalid date format. Please use dd/mm/yyyy.")
            
    category = input("Category: ").strip()

    # Input validation for amount
    while True:
        amount = input("Amount: ").strip()
        if amount.replace(".", "", 1).isdigit():
            break
        else:
            print("Enter a valid number")

    notes = input("Notes: ").strip()

    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([date, category, amount, notes])

    print("Expense added successfully!")


def load_data():
    df = pd.read_csv(csv_file)

    if df.empty or len(df) <= 1: # Check for empty or only header
        return pd.DataFrame() # Return an empty DataFrame consistently
        
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
    
    # Drop rows where date or amount couldn't be parsed
    df = df.dropna(subset=["date", "amount"])

    return df

def view_expenses():
    df = load_data()

    if df.empty:
        print("No expenses recorded yet!")
        return

    print("\nAll expenses:\n")
    # Using df.sort_values to ensure the view is chronological
    print(df.sort_values("date").to_string(index=False))

    total = df["amount"].sum()
    print(f"\nTotal expenses: {total:.2f}")


def predict_future_expenses():
    df = load_data()

    if df.empty:
        print("No data available to train the model.")
        return
        
    # Group by date to get daily total expenses
    daily = df.groupby("date")["amount"].sum().reset_index()

    if len(daily) < 2:
        print("Need at least 2 different days of data to predict.")
        return
        
    daily = daily.sort_values("date").reset_index(drop=True)
    
    # Create the X variable (number of days elapsed since the start date)
    daily["day_num"] = (daily["date"] - daily["date"].iloc[0]).dt.days

    X = daily[["day_num"]].values # Independent variable (Time)
    y = daily["amount"].values     # Dependent variable (Expense Amount)

    model = LinearRegression()
    model.fit(X, y)
    
    # Input validation for days to predict
    while True:
        days_str = input("How many future days to predict? ").strip()
        if days_str.isdigit() and int(days_str) > 0:
            days_ahead = int(days_str)
            break
        else:
            print("Enter a positive whole number (like 7 or 30).")

    # Calculate future day numbers based on the time index (day_num)
    last_day_num = daily["day_num"].iloc[-1]
    
    # last_date is not strictly needed if we use the first date and day_num
    # For simplicity, we can use the last date and timedelta
    last_date = daily["date"].iloc[-1]
    
    future_day_nums = np.array(
        [last_day_num + i for i in range(1, days_ahead + 1)]
    ).reshape(-1, 1)

    # Calculate the actual future calendar dates
    future_dates = [last_date + timedelta(days=i) for i in range(1, days_ahead + 1)]

    predicted_amounts = model.predict(future_day_nums)

    print("\nPredicted future expenses:\n")
    for d, amt in zip(future_dates, predicted_amounts):
        # Prevent predicting negative expenses, set to 0.00 if negative
        safe_amt = max(0, amt) 
        print(d.strftime("%d/%m/%Y"), "->", f"{safe_amt:.2f}")


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