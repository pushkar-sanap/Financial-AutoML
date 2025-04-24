import streamlit as st
import pandas as pd
import numpy as np
from src.models.ml_models import generate_future_dates, forecast_with_model, perform_time_series_forecast
from src.visualization.data_viz import plot_forecast, plot_time_series_forecast

def show_future_predictions_page(df, date_column):
    """Show future predictions page content"""
    st.header("Future Predictions")
    
    # Get numerical columns
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    if date_column != "None":
        # Check if a model has been trained
        if 'best_model' in st.session_state:
            st.subheader("Forecast Future Values")
            
            # Forecast settings
            forecast_periods = st.slider("Number of periods to forecast", 1, 365, 30)
            
            if st.button("Generate Forecast"):
                # Generate future dates
                future_dates = generate_future_dates(df, date_column, forecast_periods)
                
                if future_dates is not None:
                    # Time series forecasting with the best model
                    try:
                        best_model = st.session_state['best_model']
                        best_model_name = st.session_state['best_model_name']
                        feature_columns = st.session_state['feature_columns']
                        scaler = st.session_state['scaler']
                        target_column = st.session_state['target_column']
                        
                        # Generate forecast
                        historical, forecast, future_df = forecast_with_model(
                            df, best_model, feature_columns, scaler, 
                            date_column, target_column, future_dates
                        )
                        
                        # Plot forecast
                        plot_forecast(historical, forecast, date_column, target_column)
                        
                        # Display forecast data
                        st.subheader("Forecast Data")
                        st.dataframe(future_df[[date_column, target_column]])
                        
                    except Exception as e:
                        st.error(f"Error generating forecast: {e}")
            
            # Time series models
            st.subheader("Time Series Modeling")
            
            # Select column to forecast
            ts_target_column = st.selectbox(
                "Select column to forecast",
                options=num_cols,
                index=num_cols.index(st.session_state['target_column']) if 'target_column' in st.session_state and st.session_state['target_column'] in num_cols else 0
            )
            
            if st.button("Run Time Series Analysis"):
                try:
                    # Check if enough data
                    if len(df) < 10:
                        st.warning("Not enough data for time series modeling.")
                        return
                    
                    # Use the selected target column
                    target_column = ts_target_column
                    
                    # Perform time series forecast
                    ts_data, forecast = perform_time_series_forecast(df, date_column, target_column)
                    
                    # Plot time series forecast
                    plot_time_series_forecast(ts_data, forecast, target_column)
                    
                except Exception as e:
                    st.error(f"Error in time series analysis: {e}")
        else:
            st.info("Please train a model in the Modeling tab first")
    else:
        st.info("Time series forecasting requires a date column. Please select a date column in the Data Preprocessing section.") 