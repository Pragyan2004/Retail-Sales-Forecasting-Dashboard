from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  
import io
import base64
import pickle
import json
from datetime import datetime, timedelta
import warnings
import os
warnings.filterwarnings('ignore')

app = Flask(__name__)
stores_df = None
sales_df = None
weekly_total_sales = None
prophet_model = None
arima_model = None
forecast_results = None

def generate_synthetic_sales_data(stores_df, periods=365*2):
    np.random.seed(42)
    base_sales = {'A': 8000, 'B': 5000, 'C': 3000}
    size_factor = stores_df['Size'] / stores_df['Size'].mean()
    
    sales_data = []
    start_date = datetime(2022, 1, 1)
    
    for store_id, store_type, size in zip(stores_df['Store'], stores_df['Type'], stores_df['Size']):
        base = base_sales[store_type] * (size_factor[store_id-1] * 0.5 + 0.5)
        
        for day in range(periods):
            current_date = start_date + timedelta(days=day)
            day_of_week = current_date.weekday()
            month = current_date.month
            week_of_year = current_date.isocalendar().week
            daily_sales = base * np.random.normal(1, 0.1) 
            if day_of_week >= 5:  
                daily_sales *= 1.3
            else:
                daily_sales *= 0.9
            if month == 12: 
                daily_sales *= 1.5
            elif month in [6, 7]: 
                daily_sales *= 1.2
            elif month in [1, 2]:  
                daily_sales *= 0.8
            daily_sales *= np.random.normal(1, 0.05)
            daily_sales = max(daily_sales, 100)
            
            sales_data.append({
                'Store': store_id,
                'Type': store_type,
                'Date': current_date,
                'Sales': daily_sales,
                'Size': size
            })
    
    return pd.DataFrame(sales_data)

def load_or_create_data():
    """Load existing data or create synthetic data if files don't exist"""
    global stores_df, sales_df, weekly_total_sales, prophet_model, arima_model, forecast_results
    
    try:
        if os.path.exists('stores.csv'):
            stores_df = pd.read_csv('stores.csv')
            print("Stores data loaded from CSV")
        else:
            return False
        if os.path.exists('weekly_sales_processed.csv'):
            sales_df = pd.read_csv('weekly_sales_processed.csv')
            sales_df['Date'] = pd.to_datetime(sales_df['Date'])
            print("Sales data loaded from CSV")
        else:
            print("Generating synthetic sales data...")
            sales_df = generate_synthetic_sales_data(stores_df, periods=365*2)
            sales_df['Date'] = pd.to_datetime(sales_df['Date'])
            sales_df = sales_df.groupby([pd.Grouper(key='Date', freq='W'), 'Store', 'Type', 'Size']).agg({
                'Sales': 'sum'
            }).reset_index()
            sales_df.to_csv('weekly_sales_processed.csv', index=False)
            print("Synthetic sales data created and saved")
        weekly_total_sales = sales_df.groupby('Date')['Sales'].sum().reset_index()
        if os.path.exists('prophet_model.pkl'):
            with open('prophet_model.pkl', 'rb') as f:
                prophet_model = pickle.load(f)
            print("Prophet model loaded")
        else:
            print("Warning: Prophet model not found")
        
        if os.path.exists('arima_model.pkl'):
            with open('arima_model.pkl', 'rb') as f:
                arima_model = pickle.load(f)
            print("ARIMA model loaded")
        else:
            print("Warning: ARIMA model not found")
        if os.path.exists('forecast_results.json'):
            with open('forecast_results.json', 'r') as f:
                forecast_results = json.load(f)
            print("Forecast results loaded")
        else:
            forecast_results = {
                'future_predictions': {
                    'ds': [datetime.now().strftime('%Y-%m-%d')],
                    'yhat': [10000],
                    'yhat_lower': [8000],
                    'yhat_upper': [12000]
                },
                'model_metrics': {
                    'MAE': {'ARIMA': 1500, 'Prophet': 1200},
                    'RMSE': {'ARIMA': 1800, 'Prophet': 1500},
                    'MAPE': {'ARIMA': 15.5, 'Prophet': 12.3}
                },
                'last_training_date': datetime.now().strftime('%Y-%m-%d')
            }
            print("Default forecast results created")
        
        return True
        
    except Exception as e:
        print(f"Error in load_or_create_data: {e}")
        return False

def create_plot():
    """Helper function to create matplotlib plot and convert to base64"""
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', dpi=100)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url
load_or_create_data()

@app.route('/')
def index():
    """Homepage - Dashboard Overview"""
    if stores_df is None:
        return render_template('error.html', message="Data not loaded. Please check if stores.csv exists.")
    
    try:
        total_stores = len(stores_df)
        store_types = stores_df['Type'].value_counts().to_dict()
        
        if sales_df is not None:
            total_sales = sales_df['Sales'].sum()
            avg_weekly_sales = sales_df.groupby('Date')['Sales'].sum().mean()
            recent_sales_data = sales_df.groupby('Date')['Sales'].sum().tail(10)
            recent_sales = {str(date): sales for date, sales in recent_sales_data.items()}
        else:
            total_sales = 0
            avg_weekly_sales = 0
            recent_sales = {}
        
        return render_template('index.html',
                             total_stores=total_stores,
                             store_types=store_types,
                             total_sales=total_sales,
                             avg_weekly_sales=avg_weekly_sales,
                             recent_sales=recent_sales)
    except Exception as e:
        return render_template('error.html', message=f"Error loading dashboard: {str(e)}")

@app.route('/stores')
def stores_analysis():
    """Store Analysis Page"""
    if stores_df is None:
        return render_template('error.html', message="Stores data not loaded")
    
    try:
        store_stats = stores_df.groupby('Type').agg({
            'Store': 'count',
            'Size': ['mean', 'min', 'max']
        }).round(2)
        if sales_df is not None:
            sales_by_type = sales_df.groupby('Type')['Sales'].agg(['sum', 'mean', 'count']).round(2)
        else:
            sales_by_type = pd.DataFrame(columns=['sum', 'mean', 'count'])
        plt.figure(figsize=(10, 6))
        stores_df['Type'].value_counts().plot(kind='bar', color=['skyblue', 'lightcoral', 'lightgreen'])
        plt.title('Store Distribution by Type')
        plt.xlabel('Store Type')
        plt.ylabel('Number of Stores')
        plt.grid(True, alpha=0.3)
        store_dist_plot = create_plot()
        plt.figure(figsize=(10, 6))
        plt.hist(stores_df['Size'], bins=15, alpha=0.7, color='orange', edgecolor='black')
        plt.title('Store Size Distribution')
        plt.xlabel('Store Size')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        store_size_plot = create_plot()
        
        return render_template('stores.html',
                             store_stats=store_stats,
                             sales_by_type=sales_by_type,
                             store_dist_plot=store_dist_plot,
                             store_size_plot=store_size_plot)
    except Exception as e:
        return render_template('error.html', message=f"Error in stores analysis: {str(e)}")

@app.route('/sales-trends')
def sales_trends():
    """Sales Trends Analysis Page"""
    if sales_df is None:
        return render_template('error.html', message="Sales data not available")
    
    try:
        weekly_total_sales = sales_df.groupby('Date')['Sales'].sum()
        plt.figure(figsize=(12, 6))
        plt.plot(weekly_total_sales.index, weekly_total_sales.values, linewidth=2)
        plt.title('Total Weekly Sales Trend')
        plt.xlabel('Date')
        plt.ylabel('Weekly Sales ($)')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        sales_trend_plot = create_plot()
        plt.figure(figsize=(12, 6))
        store_type_sales = sales_df.groupby(['Date', 'Type'])['Sales'].sum().unstack()
        for store_type in store_type_sales.columns:
            plt.plot(store_type_sales.index, store_type_sales[store_type], label=f'Type {store_type}', linewidth=2)
        plt.title('Weekly Sales by Store Type')
        plt.xlabel('Date')
        plt.ylabel('Weekly Sales ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        store_type_plot = create_plot()
        sales_df['Month'] = sales_df['Date'].dt.month
        monthly_sales = sales_df.groupby('Month')['Sales'].mean()
        
        plt.figure(figsize=(10, 6))
        monthly_sales.plot(kind='bar', color='purple', alpha=0.7)
        plt.title('Average Monthly Sales Pattern')
        plt.xlabel('Month')
        plt.ylabel('Average Sales ($)')
        plt.grid(True, alpha=0.3)
        monthly_pattern_plot = create_plot()
        
        return render_template('sales_trends.html',
                             sales_trend_plot=sales_trend_plot,
                             store_type_plot=store_type_plot,
                             monthly_pattern_plot=monthly_pattern_plot)
    except Exception as e:
        return render_template('error.html', message=f"Error in sales trends: {str(e)}")

@app.route('/forecast')
def forecast():
    """Sales Forecast Page"""
    if forecast_results is None:
        return render_template('error.html', message="Forecast data not available")
    
    try:
        future_predictions = pd.DataFrame(forecast_results['future_predictions'])
        future_predictions['ds'] = pd.to_datetime(future_predictions['ds'])
        if sales_df is not None:
            historical_data = sales_df.groupby('Date')['Sales'].sum().reset_index()
        else:
            historical_data = pd.DataFrame({
                'Date': pd.date_range(start='2022-01-01', periods=52, freq='W'),
                'Sales': np.random.normal(10000, 2000, 52)
            })
        plt.figure(figsize=(12, 6))
        plt.plot(historical_data['Date'], historical_data['Sales'], 
                 label='Historical Sales', color='blue', linewidth=2)
        plt.plot(future_predictions['ds'], future_predictions['yhat'], 
                 label='Forecast', color='red', linewidth=2, linestyle='--')
        plt.fill_between(future_predictions['ds'], 
                        future_predictions['yhat_lower'], 
                        future_predictions['yhat_upper'], 
                        alpha=0.3, color='red', label='Confidence Interval')
        
        plt.title('Sales Forecast: Historical vs Predicted')
        plt.xlabel('Date')
        plt.ylabel('Weekly Sales ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        forecast_plot = create_plot()
        model_metrics = forecast_results['model_metrics']
        plt.figure(figsize=(10, 6))
        models = list(model_metrics['MAE'].keys())
        mae_values = [model_metrics['MAE'][model] for model in models]
        
        bars = plt.bar(models, mae_values, color=['skyblue', 'lightcoral'], alpha=0.8)
        plt.title('Model Performance Comparison (MAE)')
        plt.xlabel('Models')
        plt.ylabel('Mean Absolute Error ($)')
        plt.grid(True, alpha=0.3)
        for bar, value in zip(bars, mae_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                    f'${value:,.0f}', ha='center', va='bottom', fontweight='bold')
        
        model_comparison_plot = create_plot()
        
        return render_template('forecast.html',
                             forecast_plot=forecast_plot,
                             model_comparison_plot=model_comparison_plot,
                             future_predictions=future_predictions.to_dict('records'),
                             model_metrics=model_metrics)
    except Exception as e:
        return render_template('error.html', message=f"Error in forecast: {str(e)}")

@app.route('/inventory-recommendations')
def inventory_recommendations():
    """Inventory Recommendations Page"""
    if forecast_results is None:
        return render_template('error.html', message="Forecast data not available for inventory planning")
    
    try:
        future_predictions = pd.DataFrame(forecast_results['future_predictions'])
        
        avg_forecast = future_predictions['yhat'].mean()
        max_forecast = future_predictions['yhat'].max()
        min_forecast = future_predictions['yhat'].min()
        
        safety_stock = avg_forecast * 0.2  
        reorder_point = avg_forecast + safety_stock
        optimal_order_quantity = avg_forecast * 1.5  
        
        seasonal_factors = {
            'December': 1.5,  
            'June': 1.2,     
            'July': 1.2,    
            'January': 0.8,   
            'February': 0.8   
        }
        
        if sales_df is not None:
            store_type_analysis = sales_df.groupby('Type').agg({
                'Sales': ['mean', 'std', 'count']
            }).round(2)
            store_type_analysis_dict = store_type_analysis.to_dict()
        else:
            store_type_analysis_dict = {}
        plt.figure(figsize=(12, 6))
        
        weeks = range(1, len(future_predictions) + 1)
        plt.plot(weeks, future_predictions['yhat'], label='Expected Demand', linewidth=2)
        plt.axhline(y=reorder_point, color='red', linestyle='--', label='Reorder Point')
        plt.axhline(y=safety_stock, color='orange', linestyle='--', label='Safety Stock')
        
        plt.fill_between(weeks, future_predictions['yhat_lower'], future_predictions['yhat_upper'], 
                        alpha=0.3, label='Demand Uncertainty')
        
        plt.title('Inventory Planning Dashboard')
        plt.xlabel('Weeks Ahead')
        plt.ylabel('Units / Sales ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        inventory_plot = create_plot()
        
        return render_template('inventory.html',
                             avg_forecast=avg_forecast,
                             max_forecast=max_forecast,
                             min_forecast=min_forecast,
                             safety_stock=safety_stock,
                             reorder_point=reorder_point,
                             optimal_order_quantity=optimal_order_quantity,
                             seasonal_factors=seasonal_factors,
                             store_type_analysis=store_type_analysis_dict,
                             inventory_plot=inventory_plot)
    except Exception as e:
        return render_template('error.html', message=f"Error in inventory recommendations: {str(e)}")

@app.route('/api/forecast-data')
def api_forecast_data():
    """API endpoint for forecast data"""
    if forecast_results is None:
        return jsonify({'error': 'Forecast data not available'})
    
    return jsonify(forecast_results)

@app.route('/api/store-performance')
def api_store_performance():
    """API endpoint for store performance data"""
    if sales_df is None:
        return jsonify({'error': 'Sales data not available'})
    
    try:
        store_performance = sales_df.groupby('Store').agg({
            'Sales': ['sum', 'mean', 'std'],
            'Type': 'first'
        }).round(2)
        
        return jsonify(store_performance.to_dict())
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/reload-data')
def reload_data():
    """Reload data endpoint"""
    success = load_or_create_data()
    if success:
        return jsonify({'status': 'success', 'message': 'Data reloaded successfully'})
    else:
        return jsonify({'status': 'error', 'message': 'Failed to reload data'})

if __name__ == '__main__':
    print("Starting Retail Sales Forecasting Dashboard...")
    print("Make sure you have 'stores.csv' in the same directory")
    app.run(debug=True, host='0.0.0.0', port=5000)