# TCS Stock Prediction Using LSTM (Deep Learning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 📥 Load the dataset
df = pd.read_csv('TCS_stock_history.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# 📈 Use only 'Close' prices for LSTM
close_data = df[['Date', 'Close']].copy()

# 🔁 Normalize the close prices between 0 and 1
scaler = MinMaxScaler()
close_data['Close_Scaled'] = scaler.fit_transform(close_data[['Close']])

# 🧠 Create Sequences of 60 days
sequence_length = 60
X_lstm = []
y_lstm = []
scaled_close = close_data['Close_Scaled'].values

for i in range(sequence_length, len(scaled_close)):
    X_lstm.append(scaled_close[i-sequence_length:i])
    y_lstm.append(scaled_close[i])

X_lstm = np.array(X_lstm)
y_lstm = np.array(y_lstm)
X_lstm = np.reshape(X_lstm, (X_lstm.shape[0], X_lstm.shape[1], 1))

# 🧱 Build the LSTM Model
model = Sequential()
model.add(LSTM(units=50, return_sequences=False, input_shape=(X_lstm.shape[1], 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# 🏋️ Train the Model
model.fit(X_lstm, y_lstm, epochs=20, batch_size=32)

# 🔮 Predict the Next Close Price
last_60_days = scaled_close[-60:]
X_test = np.reshape(last_60_days, (1, 60, 1))
predicted_scaled = model.predict(X_test)
predicted_price = scaler.inverse_transform(predicted_scaled)

# 📢 Print Predicted Price
print(f"\n📈 Predicted Next Close Price: ₹{predicted_price[0][0]:.2f}")

# 📊 Plot the Prediction vs Historical Data
plt.figure(figsize=(12, 6))
plt.plot(close_data['Date'], close_data['Close'], label="Historical Close Prices", color='blue')
plt.axhline(y=predicted_price[0][0], color='red', linestyle='--', label=f"Predicted Next Price ₹{predicted_price[0][0]:.2f}")
plt.title("TCS Close Price + LSTM Prediction")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
