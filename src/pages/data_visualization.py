import streamlit as st
import pandas as pd
import numpy as np
from src.visualization.data_viz import (
    plot_time_series,
    plot_scatter,
    plot_categorical_analysis
)

def show_data_visualization_page(df, date_column):
    """Show data visualization page content"""
    st.header("Data Visualization")
    
    # Get numerical and categorical columns
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Time series visualization if date column exists
    if date_column != "None":
        ts_cols = st.multiselect(
            "Select numerical columns for time series visualization",
            options=num_cols,
            default=num_cols[:min(2, len(num_cols))]
        )
        
        plot_time_series(df, date_column, ts_cols)
    
    # Scatter plot
    st.subheader("Scatter Plot Analysis")
    if len(num_cols) >= 2:
        x_col = st.selectbox("Select X-axis column", options=num_cols)
        y_col = st.selectbox(
            "Select Y-axis column", 
            options=[col for col in num_cols if col != x_col]
        )
        
        plot_scatter(df, x_col, y_col)
    
    # Category visualization
    if cat_cols:
        st.subheader("Categorical Data Analysis")
        cat_col = st.selectbox("Select categorical column", options=cat_cols)
        
        plot_categorical_analysis(df, cat_col, num_cols) 