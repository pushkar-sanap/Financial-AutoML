import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose

def visualize_data(df, date_column, tab):
    """
    Performs and displays data visualizations based on the selected tab.

    Args:
        df (pandas.DataFrame): The DataFrame to visualize.
        date_column (str): The name of the date column (or None).
        tab (int):  1 for EDA tab, 2 for Visualization tab.
    """

    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    if tab == 1:
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
        selected_cols = st.multiselect(
            "Select columns for distribution analysis",
            options=num_cols,
            default=num_cols[:min(5, len(num_cols))]
        )

        if selected_cols:
            for col in selected_cols:
                fig = px.histogram(df, x=col, title=f"Distribution of {col}")
                st.plotly_chart(fig)

    elif tab == 2:
        st.markdown('<h1 class="main-header">Data Visualization</h1>', unsafe_allow_html=True)

        # Time series visualization if date column exists
        if date_column != "None":
            st.markdown('<h2 class="section-header">Time Series Analysis</h2>', unsafe_allow_html=True)

            ts_cols = st.multiselect(
                "Select numerical columns for time series visualization",
                options=num_cols,
                default=num_cols[:min(2, len(num_cols))]
            )

            if ts_cols:
                for col in ts_cols:
                    st.markdown(f'<h3 class="subsection-header">Analysis of {col}</h3>', unsafe_allow_html=True)
                    fig = px.line(df, x=date_column, y=col, title=f"Time Series of {col}")
                    st.plotly_chart(fig)

                    # Seasonal decomposition
                    st.subheader(f"Seasonal Decomposition of {col}")
                    try:
                        decomposition = seasonal_decompose(df.set_index(date_column)[col].dropna(), model='additive')
                        fig_seasonal = go.Figure()

                        fig_seasonal.add_trace(go.Scatter(x=decomposition.observed.index, y=decomposition.observed, name='Observed'))
                        fig_seasonal.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend, name='Trend'))
                        fig_seasonal.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, name='Seasonal'))
                        fig_seasonal.add_trace(go.Scatter(x=decomposition.resid.index, y=decomposition.resid, name='Residual'))

                        fig_seasonal.update_layout(title=f'Seasonal Decomposition of {col}', showlegend=True)
                        st.plotly_chart(fig_seasonal)
                    except Exception as e:
                        st.error(f"Error in seasonal decomposition for {col}: {e}")

        # Scatter plots
        st.subheader("Scatter Plots")
        scatter_x = st.selectbox("Select X-axis for scatter plot", options=num_cols, key="scatter_x")
        scatter_y = st.selectbox("Select Y-axis for scatter plot", options=num_cols, key="scatter_y")

        if scatter_x and scatter_y and scatter_x != scatter_y:
            fig_scatter = px.scatter(df, x=scatter_x, y=scatter_y, title=f"Scatter Plot of {scatter_x} vs {scatter_y}")
            st.plotly_chart(fig_scatter)

        # Box plots
        st.subheader("Box Plots")
        box_cols = st.multiselect("Select columns for box plots", options=num_cols, default=num_cols[:min(3, len(num_cols))])
        if box_cols:
            for col in box_cols:
                fig_box = px.box(df, y=col, title=f"Box Plot of {col}")
                st.plotly_chart(fig_box)