# TCS Stock Price Analysis using Linear Regression

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# 📥 Load dataset
df = pd.read_csv('TCS_stock_history.csv')

# 🕒 Convert 'Date' column to datetime and sort
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by='Date')

# 🧹 Check for missing values and fill them
print("Missing values:\n", df.isnull().sum())
df.ffill(inplace=True)

# 📊 Show basic info
print("\nDataset Info:")
print(df.info())
print(df.head())

# 📈 Plot Close Price Over Time
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Close'], label='Close Price', color='blue')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('TCS Close Price Over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 🧠 Feature Engineering
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day_of_Week'] = df['Date'].dt.dayofweek
df['Prev_Close'] = df['Close'].shift(1)
df.dropna(inplace=True)

# 🎯 Prepare Features and Target
# ==============================
# Linear Regression Model
# ==============================
X = df[['Open', 'High', 'Low', 'Volume', 'Prev_Close', 'Day_of_Week', 'Month']]
y = df['Close']

# 🧪 Split into Train and Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🤖 Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# 📈 Predict and Evaluate
y_pred = model.predict(X_test)
print("\n✅ Linear Regression Results:")
print("R² Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# 📉 Plot Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.xlabel('Actual Close Price')
plt.ylabel('Predicted Close Price')
plt.title('Actual vs Predicted Close Price')
plt.grid(True)
plt.tight_layout()
plt.show()

# 💾 Save the model (optional)
with open("models/TCS_Linear_Model.pkl",'wb') as f:
    pickle.dump(model, f)

# ✅ Completion message
print("\n📦 Model saved as 'TCS_Linear_Model.pkl'")
print("📈 You can now move on to the LSTM deep learning model.")
