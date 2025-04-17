import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import utility modules
from src.utils.data_processing import (
    load_data, 
    convert_date_column, 
    handle_missing_values, 
    display_missing_values, 
    create_sample_data
)

# Import page modules
from src.pages.data_analysis import show_data_analysis_page
from src.pages.data_visualization import show_data_visualization_page
from src.pages.modeling import show_modeling_page
from src.pages.future_predictions import show_future_predictions_page

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
        df = load_data(uploaded_file)
        
        if df is not None:
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
            df = convert_date_column(df, date_column)
            
            # Handle missing values
            if display_missing_values(df):
                missing_strategy = st.sidebar.selectbox(
                    "Handle missing values",
                    options=["None", "Drop", "Fill with mean", "Fill with median", "Fill with 0"]
                )
                
                if missing_strategy != "None":
                    df = handle_missing_values(df, missing_strategy)
            
            # Tabs for different sections
            tab1, tab2, tab3, tab4 = st.tabs(["Data Analysis", "Data Visualization", "Modeling", "Future Predictions"])
            
            with tab1:
                show_data_analysis_page(df)
                
            with tab2:
                show_data_visualization_page(df, date_column)
                
            with tab3:
                show_modeling_page(df)
            
            with tab4:
                show_future_predictions_page(df, date_column)
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
        sample_df = create_sample_data()
        st.dataframe(sample_df)

# Run the app
if __name__ == "__main__":
    run_automl_app() 