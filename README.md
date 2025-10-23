# 🛍️ Retail Sales Forecasting Dashboard

A comprehensive **web application for retail sales forecasting and inventory optimization** using machine learning models.


---

## 📊 Overview

This application provides retail businesses with powerful **sales forecasting capabilities** using time series analysis and machine learning.  
It helps optimize inventory management, identify sales trends, and make data-driven decisions through an intuitive **Flask-based web interface**.

---

## ✨ Features

### 🏠 Dashboard Overview
- Key performance indicators and metrics  
- Store distribution analytics  
- Recent sales performance tracking  
- Quick access to all features  

<img width="1369" height="946" alt="Screenshot 2025-10-23 190634" src="https://github.com/user-attachments/assets/5b5e06a7-c7e1-4e39-9807-80ac91d0cfcf" />

<img width="1317" height="728" alt="Screenshot 2025-10-23 190702" src="https://github.com/user-attachments/assets/5adc08a7-35e0-4996-9319-7b3df2891ecf" />

---

### 🏪 Store Analysis
- Store type distribution visualization  
- Size analysis and statistics  
- Performance metrics by store type  
- Comparative analytics  

<img width="1308" height="940" alt="Screenshot 2025-10-23 190644" src="https://github.com/user-attachments/assets/b26d6389-cfd8-44da-a384-e642e3cc3b10" />

---

### 📈 Sales Trends
- Historical sales trend visualization  
- Store type comparison over time  
- Monthly seasonal patterns  
- Interactive time series charts  

<img width="1332" height="888" alt="Screenshot 2025-10-23 190655" src="https://github.com/user-attachments/assets/52ac3b8b-f4c5-49d7-a38d-ef9d6f2881f0" />
<img width="1381" height="831" alt="Screenshot 2025-10-23 190713" src="https://github.com/user-attachments/assets/4c92e9ca-45d2-4f10-8fb1-0d41dc6dd313" />

---

### 🔮 Sales Forecasting
- 12-week sales predictions using **Prophet** and **ARIMA** models  
- Confidence interval visualization  
- Model performance comparison  
- Interactive forecast charts  


---

### 📦 Inventory Recommendations
- Optimal inventory level calculations  
- Safety stock and reorder point recommendations  
- Seasonal adjustment factors  
- Store-type specific strategies  

---

## 🛠️ Technology Stack

### Backend
- **Flask** – Web framework  
- **Pandas** – Data manipulation  
- **NumPy** – Numerical computing  
- **Matplotlib** – Data visualization  

### Machine Learning
- **Prophet** – Time series forecasting  
- **ARIMA** – Statistical forecasting model  
- **Scikit-learn** – Model evaluation metrics  

### Frontend
- **Bootstrap 5** – Responsive UI framework  
- **Chart.js** – Interactive charts  
- **Font Awesome** – Icons  

---

## 📋 Prerequisites

- Python 3.8 or higher  
- `pip` (Python package manager)

---
##  Prepare Data Files

Ensure you have the following files in the project root:

stores.csv – Store information (provided)

(Optional) Pre-trained model files from Jupyter notebook



## 🚀 Installation

###  Clone the Repository
```bash
git clone https://github.com/pragyan2004/retail-sales-forecasting.git
cd retail-sales-forecasting
```

### Prepare Data Files

Ensure you have the following files in the project root:

stores.csv – Store information (provided)

(Optional) Pre-trained model files from Jupyter notebook
---

## 🎯 Usage
### 1. Start the Application
    python app.py

### 2. Access the Dashboard

Open your browser and go to:

    http://localhost:5000

### 3. Navigate Through Pages

Home – Dashboard overview with key metrics

Stores – Store distribution and analysis

Sales Trends – Historical sales patterns

Forecast – Future sales predictions

Inventory – Stock optimization recommendations

---

### 📊 Data Processing Pipeline

 Data Loading & Preprocessing

    Load store information from CSV

    Generate synthetic sales data (if needed)

    Handle missing values and validation

Time Series Analysis

    Weekly aggregation of sales

    Seasonal decomposition

    Trend identification

    Stationarity testing

Model Training

    Prophet and ARIMA models

    Evaluate with MAE, RMSE, MAPE

Forecasting & Visualization

    12-week predictions

    Confidence intervals

    Interactive charts

---

## 🧠 Machine Learning Models

### Prophet Model

Handles seasonality and holidays

Robust to missing data

Automatic changepoint detection

### ARIMA Model

Statistical time series analysis

Handles trend and seasonality

Configurable parameters (p, d, q)

### Model Evaluation Metrics

MAE – Mean Absolute Error

RMSE – Root Mean Square Error

MAPE – Mean Absolute Percentage Error

---

## 📈 Business Insights

### Inventory Optimization

Safety stock and reorder point recommendations

Seasonal adjustments

Store-type strategies

### Sales Predictions

Weekly forecasts

Confidence intervals

Trend and seasonality insights

### Store Performance

Type-based analytics

Size correlation analysis

Performance benchmarking

Growth opportunities


