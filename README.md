# xgboost-stock-prediction-framework
A general-purpose stock price forecasting tool using XGBoost. Upload any stock’s historical data (8+ years recommended) to generate forecasts, performance metrics, and visualizations. Includes data cleaning, feature engineering, and backtesting.
##🚀 Features
* General-Purpose: Works with any stock’s historical data (minimum 8 years recommended).
* Data Preprocessing: Cleans and prepares raw stock price data for analysis.
* Feature Engineering: Computes log returns and applies smoothing to reduce noise.
* Supervised Learning Transformation: Converts time series data into a supervised format suitable for machine learning.
* Model Training & Backtesting: Trains an XGBoost regressor, evaluates on a holdout set, and reports robust metrics (RMSE, MAE, MAPE, R²).
* Forecasting: Predicts future stock prices for a user-specified period.
* Visualization: Generates high-quality plots for training, testing, and forecast periods.
* Detailed Reporting: Prints tabular and graphical summaries for easy interpretation.

## 📁 File Structure
* xgboost_comprehensive.py — Main script for data processing, modeling, and visualization.
* ICICIBANK_NS_DAILY_2015_2025.csv — Historical stock price data (you can download any stock price data from yfinance).
* requirements.txt — Python dependencies.

## 📊 Example Outputs
* Training/test split plot with predictions
* Actual vs. predicted prices for test and forecast periods
* Tabular summary of forecast accuracy
* Automated performance metrics and visualizations

## 🛠️ Getting Started
### Prepare Your Data
   * Add your historical stock price CSV file to the project directory.
   * Required columns: Date, Close (additional columns are ignored).
   * Date format: YYYY-MM-DD
   * Recommendation: Use stocks with at least 8 years of data for best results.
### Configure the Script
   * Open xgboost_comprehensive.py.
   * Update the filename in the line:
     * df = pd.read_csv("your_stock_data.csv")
   * Optionally, adjust the training and forecast periods at the top of the script.

###  View Results
   * The script prints evaluation metrics and displays several plots for model performance and forecasts.
   * Detailed tabular and graphical summaries are provided for both backtesting and forecast periods.

### 📈 Metrics & Evaluation
   * RMSE (Root Mean Squared Error)
   * MAE (Mean Absolute Error)
   * MAPE (Mean Absolute Percentage Error)
   * R² Score
   * All metrics are reported for both price and log returns, on test and forecast periods.
     
