import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pandas as pd

def get_available_models(random_state=42):
    """Return dictionary of available models"""
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Lasso Regression": Lasso(),
        "Random Forest": RandomForestRegressor(random_state=random_state),
        "Gradient Boosting": GradientBoostingRegressor(random_state=random_state),
        "XGBoost": xgb.XGBRegressor(random_state=random_state)
    }
    return models

def train_and_evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test, selected_models, models):
    """Train and evaluate selected models"""
    results = []
    trained_models = {}
    
    for model_name in selected_models:
        st.text(f"Training {model_name}...")
        model = models[model_name]
        model.fit(X_train_scaled, y_train)
        trained_models[model_name] = model
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results.append({
            'Model': model_name,
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2 Score': r2
        })
    
    return pd.DataFrame(results), trained_models

def find_best_model(results_df):
    """Find the best model based on R2 score"""
    best_model_name = results_df.loc[results_df['R2 Score'].idxmax()]['Model']
    best_r2 = results_df['R2 Score'].max()
    return best_model_name, best_r2

def generate_future_dates(df, date_column, forecast_periods):
    """Generate future dates based on time series frequency"""
    last_date = df[date_column].max()
    
    # Generate future dates
    if pd.api.types.is_datetime64_any_dtype(df[date_column]):
        # Determine the frequency of the data
        if len(df) >= 2:
            # Calculate the most common difference
            date_diffs = df[date_column].sort_values().diff().dropna()
            if not date_diffs.empty:
                most_common_diff = date_diffs.value_counts().idxmax()
                future_dates = [last_date + i * most_common_diff for i in range(1, forecast_periods + 1)]
            else:
                # Default to daily frequency
                future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, forecast_periods + 1)]
        else:
            # Default to daily frequency
            future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, forecast_periods + 1)]
        
        return future_dates
    else:
        st.error("Date column is not in datetime format.")
        return None

def forecast_with_model(df, best_model, feature_columns, scaler, date_column, target_column, future_dates):
    """Generate forecast using best ML model"""
    # Create a dataframe for future predictions
    future_df = pd.DataFrame({date_column: future_dates})
    
    # Generate future features (using last n values average)
    n_last = min(30, len(df))
    for feature in feature_columns:
        if feature in df.columns:
            # Use average of last n values as future value
            future_df[feature] = df[feature].tail(n_last).mean()
    
    # Scale the features
    X_future = future_df[feature_columns]
    X_future_scaled = scaler.transform(X_future)
    
    # Make predictions
    future_predictions = best_model.predict(X_future_scaled)
    
    # Add predictions to future dataframe
    future_df[target_column] = future_predictions
    
    # Format dataframes for visualization
    historical = df[[date_column, target_column]].copy()
    historical['Type'] = 'Historical'
    
    forecast = future_df[[date_column, target_column]].copy()
    forecast['Type'] = 'Forecast'
    
    return historical, forecast, future_df

def perform_time_series_forecast(df, date_column, target_column, periods=30):
    """Perform time series forecasting using Holt-Winters method"""
    # Prepare the time series data
    ts_data = df.set_index(date_column)[target_column]
    
    # Simple Exponential Smoothing with trend and seasonality
    model = ExponentialSmoothing(
        ts_data,
        trend='add',
        seasonal='add',
        seasonal_periods=7
    ).fit()
    
    # Forecast
    forecast_periods = min(periods, len(ts_data) // 2)
    forecast = model.forecast(forecast_periods)
    
    return ts_data, forecast 