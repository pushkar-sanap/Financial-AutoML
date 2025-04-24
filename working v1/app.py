import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Financial AutoML",
    page_icon="💰",
    layout="wide"
)

# App title and description
st.title("Financial AutoML Analysis")
st.markdown("""
This app performs automated machine learning on financial data.
Upload your financial CSV file to get insights and predictions!
""")

# Sidebar configuration
st.sidebar.header("Configuration")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload your financial CSV file", type=["csv"])

# Main function to run the app
def run_automl_app():
    if uploaded_file is not None:
        # Load data
        try:
            df = pd.read_csv(uploaded_file)
            st.success("Data loaded successfully!")
            
            # Display raw data
            with st.expander("Raw Data Preview"):
                st.dataframe(df)
                st.write(f"Shape of the dataset: {df.shape}")
            
            # Data preprocessing options
            st.sidebar.subheader("Data Preprocessing")
            date_column = st.sidebar.selectbox(
                "Select date column (if any)",
                options=["None"] + list(df.columns),
            )
            
            # Convert date column if selected
            if date_column != "None":
                try:
                    df[date_column] = pd.to_datetime(df[date_column])
                    df = df.sort_values(by=date_column)
                    st.info(f"Converted '{date_column}' to datetime and sorted data.")
                except Exception as e:
                    st.error(f"Error converting date column: {e}")
            
            # Handle missing values
            if df.isna().sum().sum() > 0:
                st.subheader("Missing Values")
                missing_df = pd.DataFrame({
                    'Column': df.columns,
                    'Missing Values': df.isna().sum().values,
                    'Percentage': (df.isna().sum().values / len(df)) * 100
                })
                st.dataframe(missing_df)
                
                # Handle missing values
                missing_strategy = st.sidebar.selectbox(
                    "Handle missing values",
                    options=["None", "Drop", "Fill with mean", "Fill with median", "Fill with 0"]
                )
                
                if missing_strategy == "Drop":
                    df = df.dropna()
                    st.info("Dropped rows with missing values")
                elif missing_strategy == "Fill with mean":
                    for col in df.select_dtypes(include=np.number).columns:
                        df[col].fillna(df[col].mean(), inplace=True)
                    st.info("Filled numerical missing values with mean")
                elif missing_strategy == "Fill with median":
                    for col in df.select_dtypes(include=np.number).columns:
                        df[col].fillna(df[col].median(), inplace=True)
                    st.info("Filled numerical missing values with median")
                elif missing_strategy == "Fill with 0":
                    df.fillna(0, inplace=True)
                    st.info("Filled missing values with 0")
            
            # Tabs for different sections
            tab1, tab2, tab3, tab4 = st.tabs(["Data Analysis", "Data Visualization", "Modeling", "Future Predictions"])
            
            with tab1:
                st.header("Exploratory Data Analysis")
                
                # Summary statistics
                st.subheader("Summary Statistics")
                st.dataframe(df.describe())
                
                # Data types
                st.subheader("Data Types")
                data_types = pd.DataFrame({
                    'Column': df.columns,
                    'Data Type': df.dtypes.values,
                    'Non-Null Count': df.count().values,
                    'Null Count': df.isna().sum().values
                })
                st.dataframe(data_types)
                
                # Correlation heatmap
                st.subheader("Correlation Analysis")
                numeric_df = df.select_dtypes(include=np.number)
                if not numeric_df.empty:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
                    st.pyplot(fig)
                else:
                    st.info("No numerical columns for correlation analysis")
                
                # Distribution of numerical columns
                st.subheader("Distributions of Numerical Columns")
                num_cols = df.select_dtypes(include=np.number).columns.tolist()
                selected_cols = st.multiselect(
                    "Select columns for distribution analysis",
                    options=num_cols,
                    default=num_cols[:min(5, len(num_cols))]
                )
                
                if selected_cols:
                    for col in selected_cols:
                        fig = px.histogram(df, x=col, title=f"Distribution of {col}")
                        st.plotly_chart(fig)
                
            with tab2:
                st.header("Data Visualization")
                
                # Time series visualization if date column exists
                if date_column != "None":
                    st.subheader("Time Series Analysis")
                    
                    ts_cols = st.multiselect(
                        "Select numerical columns for time series visualization",
                        options=num_cols,
                        default=num_cols[:min(2, len(num_cols))]
                    )
                    
                    if ts_cols:
                        for col in ts_cols:
                            fig = px.line(df, x=date_column, y=col, title=f"{col} Over Time")
                            st.plotly_chart(fig)
                            
                            # Moving averages
                            st.subheader(f"Moving Averages for {col}")
                            df_ma = df.copy()
                            df_ma[f'{col}_MA7'] = df[col].rolling(window=7).mean()
                            df_ma[f'{col}_MA30'] = df[col].rolling(window=30).mean()
                            
                            fig = px.line(df_ma, x=date_column, 
                                          y=[col, f"{col}_MA7", f"{col}_MA30"],
                                          title=f"Moving Averages for {col}",
                                          labels={
                                              col: "Raw Values",
                                              f"{col}_MA7": "7-Day MA",
                                              f"{col}_MA30": "30-Day MA"
                                          })
                            st.plotly_chart(fig)
                            
                            # Seasonal decomposition if enough data
                            if len(df) >= 14:
                                try:
                                    st.subheader(f"Seasonal Decomposition for {col}")
                                    decomposition = seasonal_decompose(df[col], model='additive', period=7)
                                    
                                    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 12))
                                    decomposition.observed.plot(ax=ax1)
                                    ax1.set_title('Observed')
                                    decomposition.trend.plot(ax=ax2)
                                    ax2.set_title('Trend')
                                    decomposition.seasonal.plot(ax=ax3)
                                    ax3.set_title('Seasonality')
                                    decomposition.resid.plot(ax=ax4)
                                    ax4.set_title('Residuals')
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                except Exception as e:
                                    st.warning(f"Could not perform seasonal decomposition: {e}")
                
                # Scatter plot
                st.subheader("Scatter Plot Analysis")
                if len(num_cols) >= 2:
                    x_col = st.selectbox("Select X-axis column", options=num_cols)
                    y_col = st.selectbox("Select Y-axis column", options=[col for col in num_cols if col != x_col])
                    
                    fig = px.scatter(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
                    st.plotly_chart(fig)
                
                # Category visualization
                cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                if cat_cols:
                    st.subheader("Categorical Data Analysis")
                    cat_col = st.selectbox("Select categorical column", options=cat_cols)
                    
                    # Count plot - Fix the error here
                    value_counts = df[cat_col].value_counts().reset_index()
                    value_counts.columns = ['Category', 'count']
                    fig = px.bar(value_counts, x='Category', y='count', 
                                 title=f"Count of {cat_col}")
                    st.plotly_chart(fig)
                    
                    # Relation with numerical variable
                    if num_cols:
                        num_col = st.selectbox("Select numerical column for analysis with categories", 
                                              options=num_cols)
                        
                        fig = px.box(df, x=cat_col, y=num_col, 
                                     title=f"{num_col} Distribution by {cat_col}")
                        st.plotly_chart(fig)
                
            with tab3:
                st.header("Predictive Modeling")
                
                # Select target and features
                target_col = st.selectbox(
                    "Select target column for prediction",
                    options=num_cols
                )
                
                # Exclude target from feature list
                feature_cols = [col for col in num_cols if col != target_col]
                
                # Only proceed if there are features to use
                if feature_cols:
                    selected_features = st.multiselect(
                        "Select features for modeling",
                        options=feature_cols,
                        default=feature_cols
                    )
                    
                    if selected_features:
                        test_size = st.slider("Test size (%)", 10, 50, 20) / 100
                        random_state = 42
                        
                        # Prepare data for modeling
                        X = df[selected_features]
                        y = df[target_col]
                        
                        # Handle categorical variables
                        X = pd.get_dummies(X, drop_first=True)
                        
                        # Split the data
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size, random_state=random_state
                        )
                        
                        # Scale features
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        
                        # Model selection
                        models = {
                            "Linear Regression": LinearRegression(),
                            "Ridge Regression": Ridge(),
                            "Lasso Regression": Lasso(),
                            "Random Forest": RandomForestRegressor(random_state=random_state),
                            "Gradient Boosting": GradientBoostingRegressor(random_state=random_state),
                            "XGBoost": xgb.XGBRegressor(random_state=random_state)
                        }
                        
                        selected_models = st.multiselect(
                            "Select models to train",
                            options=list(models.keys()),
                            default=["Linear Regression", "Random Forest"]
                        )
                        
                        if selected_models and st.button("Train Models"):
                            st.subheader("Model Performance")
                            
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
                            
                            # Display results
                            results_df = pd.DataFrame(results)
                            st.dataframe(results_df)
                            
                            # Find best model based on R2 score
                            best_model_name = results_df.loc[results_df['R2 Score'].idxmax()]['Model']
                            st.success(f"Best model: {best_model_name} with R2 Score: {results_df['R2 Score'].max():.4f}")
                            
                            # Plot actual vs predicted for best model
                            best_model = trained_models[best_model_name]
                            y_pred_best = best_model.predict(X_test_scaled)
                            
                            fig = px.scatter(x=y_test, y=y_pred_best, 
                                            labels={'x': 'Actual', 'y': 'Predicted'},
                                            title=f'Actual vs Predicted ({best_model_name})')
                            fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], 
                                                    y=[y_test.min(), y_test.max()],
                                                    mode='lines', name='Perfect Prediction'))
                            st.plotly_chart(fig)
                            
                            # Feature importance for tree-based models
                            if best_model_name in ["Random Forest", "Gradient Boosting", "XGBoost"]:
                                st.subheader("Feature Importance")
                                
                                if hasattr(best_model, 'feature_importances_'):
                                    importance_df = pd.DataFrame({
                                        'Feature': X.columns,
                                        'Importance': best_model.feature_importances_
                                    }).sort_values('Importance', ascending=False)
                                    
                                    fig = px.bar(importance_df, x='Feature', y='Importance',
                                                title=f'Feature Importance ({best_model_name})')
                                    st.plotly_chart(fig)
                                
                            # Save best model info for predictions
                            st.session_state['best_model'] = best_model
                            st.session_state['best_model_name'] = best_model_name
                            st.session_state['feature_columns'] = X.columns.tolist()
                            st.session_state['scaler'] = scaler
                            st.session_state['target_column'] = target_col
                    else:
                        st.warning("Please select at least one feature for modeling")
                else:
                    st.warning("No numerical features available for modeling (excluding target)")
            
            with tab4:
                st.header("Future Predictions")
                
                if date_column != "None":
                    # Check if a model has been trained
                    if 'best_model' in st.session_state:
                        st.subheader("Forecast Future Values")
                        
                        # Forecast settings
                        forecast_periods = st.slider("Number of periods to forecast", 1, 365, 30)
                        
                        if st.button("Generate Forecast"):
                            # Get the last date in the dataset
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
                            else:
                                st.error("Date column is not in datetime format.")
                                return
                            
                            # Time series forecasting with the best model
                            try:
                                best_model = st.session_state['best_model']
                                best_model_name = st.session_state['best_model_name']
                                feature_columns = st.session_state['feature_columns']
                                scaler = st.session_state['scaler']
                                target_column = st.session_state['target_column']
                                
                                # For demonstration, we'll use a simple approach to generate future feature values
                                # In a real-world scenario, you'd want to use more sophisticated methods
                                
                                # Create a dataframe for future predictions
                                future_df = pd.DataFrame({date_column: future_dates})
                                
                                # Generate future features (using last n values average)
                                n_last = min(30, len(df))
                                for feature in feature_columns:
                                    if feature in df.columns:
                                        # Use average of last n values as future value
                                        # This is a simplistic approach
                                        future_df[feature] = df[feature].tail(n_last).mean()
                                
                                # Scale the features
                                X_future = future_df[feature_columns]
                                X_future_scaled = scaler.transform(X_future)
                                
                                # Make predictions
                                future_predictions = best_model.predict(X_future_scaled)
                                
                                # Add predictions to future dataframe
                                future_df[target_column] = future_predictions
                                
                                # Plot historical data and predictions
                                historical = df[[date_column, target_column]].copy()
                                historical['Type'] = 'Historical'
                                
                                forecast = future_df[[date_column, target_column]].copy()
                                forecast['Type'] = 'Forecast'
                                
                                combined = pd.concat([historical, forecast])
                                
                                fig = px.line(combined, x=date_column, y=target_column, color='Type',
                                             title=f"Historical Data and Forecast of {target_column}")
                                st.plotly_chart(fig)
                                
                                # Display forecast data
                                st.subheader("Forecast Data")
                                st.dataframe(future_df[[date_column, target_column]])
                                
                                # More advanced forecast visualization
                                fig = go.Figure()
                                
                                # Add historical data
                                fig.add_trace(go.Scatter(
                                    x=historical[date_column],
                                    y=historical[target_column],
                                    mode='lines',
                                    name='Historical',
                                    line=dict(color='blue')
                                ))
                                
                                # Add forecast
                                fig.add_trace(go.Scatter(
                                    x=forecast[date_column],
                                    y=forecast[target_column],
                                    mode='lines',
                                    name='Forecast',
                                    line=dict(color='red')
                                ))
                                
                                # Add confidence interval (simplified)
                                # In a real app, you'd calculate proper prediction intervals
                                error_margin = 0.1 * forecast[target_column].std()
                                
                                fig.add_trace(go.Scatter(
                                    x=forecast[date_column],
                                    y=forecast[target_column] + error_margin,
                                    mode='lines',
                                    name='Upper Bound',
                                    line=dict(width=0),
                                    showlegend=False
                                ))
                                
                                fig.add_trace(go.Scatter(
                                    x=forecast[date_column],
                                    y=forecast[target_column] - error_margin,
                                    mode='lines',
                                    name='Lower Bound',
                                    line=dict(width=0),
                                    fill='tonexty',
                                    fillcolor='rgba(255, 0, 0, 0.2)',
                                    showlegend=False
                                ))
                                
                                fig.update_layout(
                                    title=f'Forecast of {target_column} (Model: {best_model_name})',
                                    xaxis_title='Date',
                                    yaxis_title=target_column,
                                    hovermode='x unified'
                                )
                                
                                st.plotly_chart(fig)
                                
                            except Exception as e:
                                st.error(f"Error generating forecast: {e}")
                        
                        # Time series models
                        st.subheader("Time Series Modeling")
                        
                        # Move target column selection outside the button click handler
                        if 'target_column' not in st.session_state:
                            ts_target_column = st.selectbox(
                                "Select column to forecast",
                                options=num_cols
                            )
                        else:
                            ts_target_column = st.selectbox(
                                "Select column to forecast",
                                options=num_cols,
                                index=num_cols.index(st.session_state['target_column']) if st.session_state['target_column'] in num_cols else 0
                            )
                        
                        if st.button("Run Time Series Analysis"):
                            try:
                                # Check if enough data
                                if len(df) < 10:
                                    st.warning("Not enough data for time series modeling.")
                                    return
                                
                                # Use the selected target column
                                target_column = ts_target_column
                                
                                # Prepare the time series data
                                ts_data = df.set_index(date_column)[target_column]
                                
                                # Simple Exponential Smoothing
                                model = ExponentialSmoothing(
                                    ts_data,
                                    trend='add',
                                    seasonal='add',
                                    seasonal_periods=7
                                ).fit()
                                
                                # Forecast
                                forecast_periods = min(30, len(ts_data) // 2)
                                forecast = model.forecast(forecast_periods)
                                
                                # Plot
                                fig = go.Figure()
                                
                                # Add historical data
                                fig.add_trace(go.Scatter(
                                    x=ts_data.index,
                                    y=ts_data.values,
                                    mode='lines',
                                    name='Historical',
                                    line=dict(color='blue')
                                ))
                                
                                # Add forecast
                                fig.add_trace(go.Scatter(
                                    x=forecast.index,
                                    y=forecast.values,
                                    mode='lines',
                                    name='Forecast (Holt-Winters)',
                                    line=dict(color='red')
                                ))
                                
                                fig.update_layout(
                                    title=f'Time Series Forecast of {target_column}',
                                    xaxis_title='Date',
                                    yaxis_title=target_column,
                                    hovermode='x unified'
                                )
                                
                                st.plotly_chart(fig)
                                
                            except Exception as e:
                                st.error(f"Error in time series analysis: {e}")
                    else:
                        st.info("Please train a model in the Modeling tab first")
                else:
                    st.info("Time series forecasting requires a date column. Please select a date column in the Data Preprocessing section.")
        
        except Exception as e:
            st.error(f"Error processing data: {e}")
    else:
        # Display instructions when no file is uploaded
        st.info("👆 Please upload a CSV file to get started.")
        
        # Sample data description
        st.header("Expected Data Format")
        st.markdown("""
        Your CSV file should contain financial data with:
        - Date column (optional but recommended for time series analysis)
        - Numerical columns for financial metrics (e.g., revenue, expenses, profit)
        - Categorical columns (optional) for grouping and analysis
        
        Examples of financial data that work well with this app:
        - Personal expense tracker with dates, amounts, and categories
        - Company financial statements with revenue, expenses, and profit over time
        - Investment portfolio performance with dates and returns
        """)
        
        # Sample dataset
        st.header("Sample Dataset Preview")
        sample_data = {
            'Date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
            'Revenue': [1000, 1200, 950, 1100, 1300],
            'Expenses': [800, 750, 820, 780, 850],
            'Profit': [200, 450, 130, 320, 450],
            'Category': ['Product A', 'Product B', 'Product A', 'Product C', 'Product B']
        }
        sample_df = pd.DataFrame(sample_data)
        st.dataframe(sample_df)

# Run the app
if __name__ == "__main__":
    run_automl_app() 