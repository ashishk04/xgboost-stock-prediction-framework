# 📚 Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
import calendar

# ===== USER INPUT SECTION =====
print("🔧 CONFIGURATION SETTINGS")
print("=" * 50)

# Use default values for non-interactive run
train_end_year = 2024
forecast_start_year = 2025
forecast_start_month = 1
forecast_end_month = 2

# Create date strings for filtering using proper date handling
train_end_date = f"{train_end_year}-12-31"
forecast_start_date = f"{forecast_start_year}-{forecast_start_month:02d}-01"

# Calculate proper end date for the month
last_day = calendar.monthrange(forecast_start_year, forecast_end_month)[1]
forecast_end_date = f"{forecast_start_year}-{forecast_end_month:02d}-{last_day}"

print(f"\n✅ CONFIGURATION:")
print(f"   Training/Testing data: Until {train_end_date}")
print(f"   Forecasting period: {forecast_start_date} to {forecast_end_date}")
print("=" * 50)

# 📥 Load and clean data
df = pd.read_csv("ICICIBANK_NS_DAILY_2015_2025.csv")
df = df.drop(index=[0, 1]).reset_index(drop=True)
df.rename(columns={'Price': 'Date'}, inplace=True)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce', format='%Y-%m-%d')
df = df.dropna(subset=['Date']).set_index('Date')
df[['Close']] = df[['Close']].apply(pd.to_numeric, errors='coerce')
df = df.dropna(subset=['Close'])

# Filter to data until specified year only (for training/testing)
df_until_train = df[df.index <= train_end_date]
print(f"DataFrame rows until {train_end_year}: {len(df_until_train)}")

# Get forecasting period data for comparison
df_forecast = df[(df.index >= forecast_start_date) & (df.index <= forecast_end_date)]
print(f"Forecasting period data points: {len(df_forecast)}")

# Compute log returns for data until specified year
log_close = np.log(df_until_train['Close'])
log_returns = log_close.diff().dropna()

# Reduce noise: 3-day moving average smoothing
log_returns_smoothed = log_returns.rolling(window=3, min_periods=1).mean()

# Prepare supervised learning data
look_back = 5
def create_supervised(series, look_back=1):
    X, y = [], []
    for i in range(len(series) - look_back):
        X.append(series[i:i+look_back])
        y.append(series[i+look_back])
    return np.array(X), np.array(y)

X, y = create_supervised(log_returns_smoothed.values, look_back)
print(f"X shape: {X.shape}, y shape: {y.shape}")

# Train-test split (90/10)
train_size = int(len(X) * 0.9)
train_X, train_y = X[:train_size], y[:train_size]
test_X, test_y = X[train_size:], y[train_size:]

# Fit XGBoost model
xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
xgb_model.fit(train_X, train_y)

# Forecast log returns for test set
y_pred_logret = xgb_model.predict(test_X)

# Convert log returns forecast back to price for test set
last_train_price = df_until_train['Close'].iloc[look_back + train_size]
predicted_prices = [last_train_price]
for ret in y_pred_logret:
    predicted_prices.append(predicted_prices[-1] * np.exp(ret))
predicted_prices = predicted_prices[1:]

# Actual prices for test set
actual_prices = df_until_train['Close'].iloc[look_back + train_size + 1:look_back + train_size + 1 + len(y_pred_logret)].values

# Ensure same length for evaluation
min_len = min(len(actual_prices), len(predicted_prices))
actual_prices = actual_prices[:min_len]
predicted_prices = predicted_prices[:min_len]

def safe_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask])) * 100

# Evaluation (in ₹)
xgb_mse = mean_squared_error(actual_prices, predicted_prices)
xgb_mae = mean_absolute_error(actual_prices, predicted_prices)
xgb_rmse = math.sqrt(xgb_mse)
xgb_r2 = r2_score(actual_prices, predicted_prices)
xgb_mape = safe_mape(actual_prices, predicted_prices)

# Evaluation (on log returns)
xgb_mse_logret = mean_squared_error(test_y[:len(y_pred_logret)], y_pred_logret)
xgb_mae_logret = mean_absolute_error(test_y[:len(y_pred_logret)], y_pred_logret)
xgb_rmse_logret = math.sqrt(xgb_mse_logret)
xgb_r2_logret = r2_score(test_y[:len(y_pred_logret)], y_pred_logret)

print("\n📊 XGBoost Performance (Log Returns, in ₹):")
print(f"RMSE: ₹{xgb_rmse:.2f}")
print(f"MAE: ₹{xgb_mae:.2f}")
print(f"MAPE: {xgb_mape:.2f}%")
print(f"R² Score: {xgb_r2:.4f}")

print("\n📊 XGBoost Performance (Log Returns, direct):")
print(f"RMSE: {xgb_rmse_logret:.6f}")
print(f"MAE: {xgb_mae_logret:.6f}")
print(f"R² Score: {xgb_r2_logret:.4f}")

# Prepare indices for plotting
test_index = df_until_train.index[look_back + train_size + 1:look_back + train_size + 1 + len(predicted_prices)]
train_end_index = df_until_train.index[look_back + train_size]

# ===== IMPRESSIVE PLOT STYLE =====
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams.update({'font.size': 13, 'axes.titlesize': 18, 'axes.labelsize': 15, 'legend.fontsize': 13, 'xtick.labelsize': 12, 'ytick.labelsize': 12})

# ===== PLOT 1: Training and Test Data Split =====
plt.figure(figsize=(15, 8))
plt.title(f"Training and Test Data Split with Predictions (Until {train_end_year})", fontsize=20, fontweight='bold', pad=20, color='#22223b')

# Plot training data
train_data = df_until_train['Close'][:train_end_index]
plt.plot(train_data.index, train_data.values, label='Training Data', color='#277da1', linewidth=2.5, alpha=0.8)

# Plot test data
test_data = df_until_train['Close'][train_end_index:]
plt.plot(test_data.index, test_data.values, label='Test Data', color='#43aa8b', linewidth=2.5, alpha=0.8)

# Plot test predictions
plt.plot(test_index, predicted_prices, label='Test Predictions', color='#f3722c', linewidth=3, marker='o', markersize=5)

plt.xlabel('Date')
plt.ylabel('Stock Price (₹)')
plt.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#22223b')
plt.grid(True, alpha=0.3)
plt.tight_layout(pad=2)
plt.show()

# ===== PLOT 2: Stock Price Data Till Training End =====
plt.figure(figsize=(15, 8))
plt.title(f'ICICI Bank Stock Price Data (Till {train_end_year})', fontsize=20, fontweight='bold', pad=20, color='#22223b')

# Plot full price data till training end
plt.plot(df_until_train.index, df_until_train['Close'].values, label=f'ICICI Bank Stock Price (Till {train_end_year})', color='#3a0ca3', linewidth=2.5)

# Highlight training and test periods
train_start = df_until_train.index[0]
plt.axvspan(train_start, train_end_index, alpha=0.15, color='#277da1', label='Training Period')
plt.axvspan(train_end_index, df_until_train.index[-1], alpha=0.15, color='#43aa8b', label='Test Period')

plt.xlabel('Date')
plt.ylabel('Stock Price (₹)')
plt.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#22223b')
plt.grid(True, alpha=0.3)
plt.tight_layout(pad=2)
plt.show()

# ===== PLOT 3: Test Set Performance =====
plt.figure(figsize=(15, 8))
plt.title('Test Set Performance: Actual vs Predicted', fontsize=20, fontweight='bold', pad=20, color='#22223b')

plt.plot(test_index, actual_prices, label='Actual Test Data', color='#3a0ca3', linewidth=2.5, marker='o', markersize=6)
plt.plot(test_index, predicted_prices, label='Predicted Test Data', color='#f3722c', linewidth=2.5, marker='s', markersize=6)

# Annotate last point
plt.annotate(f"Actual: ₹{actual_prices[-1]:.2f}", (test_index[-1], actual_prices[-1]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=12, color='#3a0ca3')
plt.annotate(f"Pred: ₹{predicted_prices[-1]:.2f}", (test_index[-1], predicted_prices[-1]), textcoords="offset points", xytext=(0,-18), ha='center', fontsize=12, color='#f3722c')

plt.xlabel('Date')
plt.ylabel('Stock Price (₹)')
plt.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#22223b')
plt.grid(True, alpha=0.3)
plt.tight_layout(pad=2)
plt.show()

# ===== FORECAST SPECIFIED PERIOD =====
# Get the last look_back log returns for forecasting
last_log_returns = log_returns_smoothed.values[-look_back:]
forecast_input = last_log_returns.copy()

# Forecast specified period
forecast_days = len(df_forecast)
forecasted_log_returns_forecast = []
forecasted_prices_forecast = [df_until_train['Close'].iloc[-1]]  # Start from last price of training data

for _ in range(forecast_days):
    # Predict next log return
    next_log_ret = xgb_model.predict(forecast_input.reshape(1, -1))[0]
    forecasted_log_returns_forecast.append(next_log_ret)
    
    # Calculate next price
    next_price = forecasted_prices_forecast[-1] * np.exp(next_log_ret)
    forecasted_prices_forecast.append(next_price)
    
    # Update input for next prediction
    forecast_input = np.append(forecast_input[1:], next_log_ret)

forecasted_prices_forecast = forecasted_prices_forecast[1:]  # Remove the initial price

# Get actual prices for forecast period
actual_prices_forecast = df_forecast['Close'].values
actual_dates_forecast = df_forecast.index

# Ensure same length
min_len_forecast = min(len(actual_prices_forecast), len(forecasted_prices_forecast))
actual_prices_forecast = actual_prices_forecast[:min_len_forecast]
forecasted_prices_forecast = forecasted_prices_forecast[:min_len_forecast]
actual_dates_forecast = actual_dates_forecast[:min_len_forecast]

# ===== PLOT 4: Forecast Period vs Actual =====
plt.figure(figsize=(15, 8))
forecast_period_name = f"{forecast_start_year}-{forecast_start_month:02d}"
plt.title(f'{forecast_period_name} Forecast vs Actual', fontsize=20, fontweight='bold', pad=20, color='#22223b')

plt.plot(actual_dates_forecast, actual_prices_forecast, label=f'Actual {forecast_period_name}', color='#3a0ca3', linewidth=2.5, marker='o', markersize=7)
plt.plot(actual_dates_forecast, forecasted_prices_forecast, label=f'Forecast {forecast_period_name}', color='#f3722c', linewidth=2.5, marker='s', markersize=7)

# Annotate first and last points
plt.annotate(f"Start: ₹{actual_prices_forecast[0]:.2f}", (actual_dates_forecast[0], actual_prices_forecast[0]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=12, color='#3a0ca3')
plt.annotate(f"End: ₹{actual_prices_forecast[-1]:.2f}", (actual_dates_forecast[-1], actual_prices_forecast[-1]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=12, color='#3a0ca3')
plt.annotate(f"Pred: ₹{forecasted_prices_forecast[-1]:.2f}", (actual_dates_forecast[-1], forecasted_prices_forecast[-1]), textcoords="offset points", xytext=(0,-18), ha='center', fontsize=12, color='#f3722c')

plt.xlabel('Date')
plt.ylabel('Stock Price (₹)')
plt.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#22223b')
plt.grid(True, alpha=0.3)
plt.tight_layout(pad=2)
plt.show()

# ===== PLOT 5: Combined View - End of Training + Forecast Period =====
plt.figure(figsize=(17, 8))
plt.title(f'Combined View: End of {train_end_year} + {forecast_period_name} Forecast', fontsize=20, fontweight='bold', pad=20, color='#22223b')

# Plot last 30 days of training data
last_30_days_train = df_until_train.tail(30)
plt.plot(last_30_days_train.index, last_30_days_train['Close'].values, label=f'Actual (End {train_end_year})', color='#3a0ca3', linewidth=2.5)

# Plot forecast period
plt.plot(actual_dates_forecast, actual_prices_forecast, label=f'Actual {forecast_period_name}', color='#277da1', linewidth=2.5, marker='o', markersize=6)
plt.plot(actual_dates_forecast, forecasted_prices_forecast, label=f'Forecast {forecast_period_name}', color='#f3722c', linewidth=2.5, marker='s', markersize=6)

plt.xlabel('Date')
plt.ylabel('Stock Price (₹)')
plt.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#22223b')
plt.grid(True, alpha=0.3)
plt.tight_layout(pad=2)
plt.show()

# ===== EVALUATE FORECAST PERIOD =====
forecast_mse = mean_squared_error(actual_prices_forecast, forecasted_prices_forecast)
forecast_mae = mean_absolute_error(actual_prices_forecast, forecasted_prices_forecast)
forecast_rmse = math.sqrt(forecast_mse)
forecast_r2 = r2_score(actual_prices_forecast, forecasted_prices_forecast)
forecast_mape = safe_mape(actual_prices_forecast, forecasted_prices_forecast)

print(f"\n📈 {forecast_period_name} Forecast Performance:")
print(f"RMSE: ₹{forecast_rmse:.2f}")
print(f"MAE: ₹{forecast_mae:.2f}")
print(f"MAPE: {forecast_mape:.2f}%")
print(f"R² Score: {forecast_r2:.4f}")

# ===== FORECAST PERIOD LOG RETURNS EVALUATION =====
# Calculate actual log returns for forecast period
actual_log_close_forecast = np.log(actual_prices_forecast)
actual_log_returns_forecast = np.diff(actual_log_close_forecast)

# Predicted log returns for forecast period (already in forecasted_log_returns_forecast)
pred_log_returns_forecast = np.array(forecasted_log_returns_forecast[:len(actual_log_returns_forecast)])

# Compute metrics
forecast_rmse_logret = math.sqrt(mean_squared_error(actual_log_returns_forecast, pred_log_returns_forecast))
forecast_mae_logret = mean_absolute_error(actual_log_returns_forecast, pred_log_returns_forecast)
forecast_r2_logret = r2_score(actual_log_returns_forecast, pred_log_returns_forecast)

print(f"\n📊 {forecast_period_name} Forecast Performance (Log Returns, direct):")
print(f"RMSE: {forecast_rmse_logret:.6f}")
print(f"MAE: {forecast_mae_logret:.6f}")
print(f"R² Score: {forecast_r2_logret:.4f}")

# ===== DETAILED FORECAST PERIOD COMPARISON =====
print(f"\n📊 Detailed {forecast_period_name} Comparison:")
print("Date\t\tActual\t\tForecast\t\tDifference\t\t% Error")
print("-" * 80)
for i, (date, actual, forecast) in enumerate(zip(actual_dates_forecast, actual_prices_forecast, forecasted_prices_forecast)):
    diff = actual - forecast
    pct_error = (abs(diff) / actual) * 100
    print(f"{date.strftime('%Y-%m-%d')}\t₹{actual:.2f}\t\t₹{forecast:.2f}\t\t₹{diff:.2f}\t\t{pct_error:.2f}%") 