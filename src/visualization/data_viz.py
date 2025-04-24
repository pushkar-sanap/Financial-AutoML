import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose

def display_summary_statistics(df):
    """Display summary statistics for the DataFrame"""
    st.subheader("Summary Statistics")
    st.dataframe(df.describe())

def display_data_types(df):
    """Display data types information"""
    st.subheader("Data Types")
    data_types = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes.values,
        'Non-Null Count': df.count().values,
        'Null Count': df.isna().sum().values
    })
    st.dataframe(data_types)

def plot_correlation_heatmap(df):
    """Plot correlation heatmap for numerical columns"""
    st.subheader("Correlation Analysis")
    numeric_df = df.select_dtypes(include=np.number)
    if not numeric_df.empty:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    else:
        st.info("No numerical columns for correlation analysis")

def plot_distributions(df, columns):
    """Plot distributions for selected columns"""
    st.subheader("Distributions of Numerical Columns")
    if columns:
        for col in columns:
            fig = px.histogram(df, x=col, title=f"Distribution of {col}")
            st.plotly_chart(fig)

def plot_time_series(df, date_column, columns):
    """Plot time series for selected columns"""
    st.subheader("Time Series Analysis")
    if columns:
        for col in columns:
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

def plot_scatter(df, x_col, y_col):
    """Plot scatter plot between two columns"""
    fig = px.scatter(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
    st.plotly_chart(fig)

def plot_categorical_analysis(df, cat_col, num_cols):
    """Plot categorical analysis visualizations"""
    # Count plot
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

def plot_model_performance(results_df):
    """Plot model performance comparison"""
    fig = px.bar(results_df, x='Model', y='R2 Score', 
                 title='Model Performance Comparison (R2 Score)')
    st.plotly_chart(fig)
    
    fig = px.bar(results_df, x='Model', y='RMSE', 
                 title='Model Performance Comparison (RMSE)')
    st.plotly_chart(fig)

def plot_actual_vs_predicted(y_test, y_pred, model_name):
    """Plot actual vs predicted values"""
    fig = px.scatter(x=y_test, y=y_pred, 
                    labels={'x': 'Actual', 'y': 'Predicted'},
                    title=f'Actual vs Predicted ({model_name})')
    fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], 
                            y=[y_test.min(), y_test.max()],
                            mode='lines', name='Perfect Prediction'))
    st.plotly_chart(fig)

def plot_feature_importance(model, feature_names, model_name):
    """Plot feature importance for tree-based models"""
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(importance_df, x='Feature', y='Importance',
                    title=f'Feature Importance ({model_name})')
        st.plotly_chart(fig)

def plot_forecast(historical, forecast, date_column, target_column):
    """Plot forecast visualization"""
    # Basic forecast plot
    combined = pd.concat([historical, forecast])
    fig = px.line(combined, x=date_column, y=target_column, color='Type',
                 title=f"Historical Data and Forecast of {target_column}")
    st.plotly_chart(fig)
    
    # Advanced forecast plot with confidence intervals
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
        title=f'Forecast with Confidence Intervals',
        xaxis_title='Date',
        yaxis_title=target_column,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig)

def plot_time_series_forecast(ts_data, forecast, target_column):
    """Plot time series forecast"""
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