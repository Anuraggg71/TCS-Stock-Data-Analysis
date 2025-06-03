# 📊 TCS Stock Data – Live and Latest

> **Internship Project | Submission Date: 26/06/2025**
> *By Anurag Dewangan*

---

## 🎯 Objective

Analyze and predict **TCS stock closing prices** using:

* 📈 Machine Learning: Linear Regression
* 🤖 Deep Learning: LSTM (Long Short-Term Memory)

---

## 📁 Project Structure

| File                                  | Description                                    |
| ------------------------------------- | ---------------------------------------------- |
| `tcs_stock_analysis.py`               | Linear Regression with EDA & prediction        |
| `tcs_stock_lstm.py`                   | LSTM deep learning model for price forecasting |
| `TCS_stock_history.csv`               | Dataset (2002–2024) of TCS stock prices        |
| `TCS_Stock_Project_Presentation.pptx` | Final project presentation                     |
| `README.md` / `README.txt`            | Project summary and documentation              |
| `.png` graph files                    | Visual results and model outputs               |

---

## 📊 Dataset Overview

* Source: Provided via Google Drive
* Time Range: 2002 – 2024
* Rows: 4,463
* Columns: Date, Open, High, Low, Close, Volume, Dividends, Stock Splits

---

## 🧪 Models Used

### 🔹 1. Linear Regression (ML)

* **Features:** Open, High, Low, Volume, Prev\_Close, Day\_of\_Week, Month
* **R² Score:** \~0.9999
* **MSE:** \~42
* **Output:** Close price prediction and scatter plot of actual vs predicted

### 🔹 2. LSTM (Deep Learning)

* **Data Used:** Last 60 days of scaled Close prices
* **Model:** 50 LSTM units + Dense layer
* **Output:** Predicted next close price (1-step)

> 📈 **Predicted Price:** ₹3861.72

---

## 📸 Visual Results

### 📍 Close Price Over Time

### 📍 Actual vs Predicted (Linear Regression)

### 📍 LSTM Prediction

---

## 🛠️ Technologies Used

* Python 3.11
* pandas, matplotlib, seaborn
* scikit-learn
* TensorFlow / Keras

---

## ✅ Conclusion

* Built and evaluated two models for stock prediction
* Achieved excellent accuracy using Linear Regression
* LSTM adds deep learning forecasting for future trends
* Fully documented with visualizations and presentation

---

## 📬 Contact

For queries or improvements, reach me at:
📧 [anuragdewangan1209@gmail.com](mailto:anuragdewangan1209@gmail.com)
