import streamlit as st
import pandas as pd
import numpy as np
from src.visualization.data_viz import (
    display_summary_statistics, 
    display_data_types, 
    plot_correlation_heatmap, 
    plot_distributions
)

def show_data_analysis_page(df):
    """Show data analysis page content"""
    st.header("Exploratory Data Analysis")
    
    # Summary statistics
    display_summary_statistics(df)
    
    # Data types
    display_data_types(df)
    
    # Correlation heatmap
    plot_correlation_heatmap(df)
    
    # Distribution of numerical columns
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    selected_cols = st.multiselect(
        "Select columns for distribution analysis",
        options=num_cols,
        default=num_cols[:min(5, len(num_cols))]
    )
    
    plot_distributions(df, selected_cols) 