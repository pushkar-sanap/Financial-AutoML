import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import warnings
import io
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import base64
warnings.filterwarnings('ignore')

# Function to generate PDF report
def generate_pdf_report(df, date_column, numeric_columns, financial_ratios=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30
    )
    elements.append(Paragraph("Financial Analysis Report", title_style))
    elements.append(Spacer(1, 12))
    
    # Summary Statistics
    elements.append(Paragraph("Summary Statistics", styles['Heading2']))
    elements.append(Spacer(1, 12))
    
    # Convert summary statistics to table
    summary_stats = df.describe()
    summary_data = [['Statistic'] + list(summary_stats.columns)]
    for idx in summary_stats.index:
        summary_data.append([idx] + [f"{val:.2f}" for val in summary_stats.loc[idx]])
    
    summary_table = Table(summary_data)
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 20))
    
    # Data Types
    elements.append(Paragraph("Data Types", styles['Heading2']))
    elements.append(Spacer(1, 12))
    
    data_types = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes.values,
        'Non-Null Count': df.count().values,
        'Null Count': df.isna().sum().values
    })
    
    dtype_data = [['Column', 'Data Type', 'Non-Null Count', 'Null Count']]
    for _, row in data_types.iterrows():
        dtype_data.append([str(val) for val in row])
    
    dtype_table = Table(dtype_data)
    dtype_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(dtype_table)
    elements.append(Spacer(1, 20))
    
    # Correlation Analysis
    if len(numeric_columns) > 0:
        elements.append(Paragraph("Correlation Analysis", styles['Heading2']))
        elements.append(Spacer(1, 12))
        
        # Create correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[numeric_columns].corr(), annot=True, cmap='coolwarm')
        plt.title("Correlation Heatmap")
        
        # Save heatmap to buffer
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        
        # Add heatmap to PDF
        elements.append(Image(img_buffer, width=6*inch, height=4*inch))
        elements.append(Spacer(1, 20))
    
    # Financial Ratios (if provided)
    if financial_ratios:
        elements.append(Paragraph("Financial Ratios", styles['Heading2']))
        elements.append(Spacer(1, 12))
        
        ratio_data = [['Ratio', 'Value', 'Interpretation']]
        for ratio_name, ratio_info in financial_ratios.items():
            ratio_data.append([
                ratio_name,
                f"{ratio_info['value']:.2f}",
                ratio_info['interpretation']
            ])
        
        ratio_table = Table(ratio_data)
        ratio_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(ratio_table)
    
    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer

# Set page configuration - Must be the first Streamlit command
st.set_page_config(
    page_title="Financial AutoML",
    page_icon="💰",
    layout="wide"
)

# Custom CSS for better header visibility
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        margin-bottom: 1rem !important;
        color: #1E88E5 !important;
    }
    .section-header {
        font-size: 2rem !important;
        font-weight: 600 !important;
        margin: 2rem 0 1rem 0 !important;
        color: #2C3E50 !important;
        padding-bottom: 0.5rem !important;
        border-bottom: 2px solid #1E88E5 !important;
    }
    .subsection-header {
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        margin: 1.5rem 0 1rem 0 !important;
        color: #34495E !important;
    }
    .plot-title {
        font-size: 1.2rem !important;
        font-weight: 500 !important;
        margin-bottom: 0.5rem !important;
        color: #2C3E50 !important;
    }
    </style>
""", unsafe_allow_html=True)

# App title and description
st.title("Financial AutoML Analysis")
st.markdown("""
This app performs automated machine learning on financial data.
Upload your financial CSV file to get insights and predictions!
""")

# Sidebar configuration
st.sidebar.header("Configuration")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload your financial CSV or Excel file", type=["csv", "xlsx", "xls"])

# Main function to run the app
def run_automl_app():
    if uploaded_file is not None:
        # Load data
        try:
            # Check file extension and read accordingly
            file_extension = uploaded_file.name.split('.')[-1].lower()
            if file_extension == 'csv':
                df = pd.read_csv(uploaded_file)
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file format. Please upload a CSV or Excel file.")
                return
                
            st.success("Data loaded successfully!")
            
            # Display raw data
            with st.expander("Raw Data Preview"):
                st.dataframe(df)
                st.write(f"Shape of the dataset: {df.shape}")
            
            # Data preprocessing options
            st.sidebar.subheader("Data Preprocessing")
            
            # Identify date columns automatically
            date_columns = []
            for col in df.columns:
                try:
                    pd.to_datetime(df[col])
                    date_columns.append(col)
                except:
                    continue
            
            # Date column selection with auto-detection
            date_column = st.sidebar.selectbox(
                "Select date column (if any)",
                options=["None"] + date_columns,
                help="Automatically detected date columns are shown first"
            )
            
            # Convert date column if selected
            if date_column != "None":
                try:
                    df[date_column] = pd.to_datetime(df[date_column])
                    df = df.sort_values(by=date_column)
                    st.info(f"Converted '{date_column}' to datetime and sorted data.")
                except Exception as e:
                    st.error(f"Error converting date column: {e}")
            
            # Identify numeric columns automatically
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
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
                    options=["None", "Drop", "Fill with mean", "Fill with median", "Fill with 0", "Forward fill", "Backward fill"]
                )
                
                if missing_strategy == "Drop":
                    df = df.dropna()
                    st.info("Dropped rows with missing values")
                elif missing_strategy == "Fill with mean":
                    for col in numeric_columns:
                        df[col].fillna(df[col].mean(), inplace=True)
                    st.info("Filled numerical missing values with mean")
                elif missing_strategy == "Fill with median":
                    for col in numeric_columns:
                        df[col].fillna(df[col].median(), inplace=True)
                    st.info("Filled numerical missing values with median")
                elif missing_strategy == "Fill with 0":
                    df.fillna(0, inplace=True)
                    st.info("Filled missing values with 0")
                elif missing_strategy == "Forward fill":
                    df.fillna(method='ffill', inplace=True)
                    st.info("Filled missing values using forward fill")
                elif missing_strategy == "Backward fill":
                    df.fillna(method='bfill', inplace=True)
                    st.info("Filled missing values using backward fill")
            
            # Data cleaning options
            st.sidebar.subheader("Data Cleaning")
            
            # Remove duplicates
            if st.sidebar.checkbox("Remove duplicate rows"):
                initial_rows = len(df)
                df = df.drop_duplicates()
                removed_rows = initial_rows - len(df)
                if removed_rows > 0:
                    st.info(f"Removed {removed_rows} duplicate rows")
            
            # Handle outliers
            if st.sidebar.checkbox("Handle outliers"):
                outlier_method = st.sidebar.selectbox(
                    "Outlier detection method",
                    ["IQR", "Z-score", "Percentile"]
                )
                
                if outlier_method == "IQR":
                    for col in numeric_columns:
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        df[col] = df[col].clip(lower=Q1 - 1.5*IQR, upper=Q3 + 1.5*IQR)
                elif outlier_method == "Z-score":
                    for col in numeric_columns:
                        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                        df[col] = df[col].mask(z_scores > 3, df[col].mean())
                elif outlier_method == "Percentile":
                    percentile = st.sidebar.slider("Percentile threshold", 90, 99, 95)
                    for col in numeric_columns:
                        lower = df[col].quantile((100-percentile)/100)
                        upper = df[col].quantile(percentile/100)
                        df[col] = df[col].clip(lower=lower, upper=upper)
                
                st.info(f"Handled outliers using {outlier_method} method")
            
            # Data transformation
            st.sidebar.subheader("Data Transformation")
            
            # Log transformation for highly skewed numeric columns
            if st.sidebar.checkbox("Apply log transformation to highly skewed columns"):
                for col in numeric_columns:
                    if df[col].min() > 0:  # Only apply log transform to positive values
                        skewness = df[col].skew()
                        if abs(skewness) > 1:  # Apply if highly skewed
                            df[f"{col}_log"] = np.log1p(df[col])
                            st.info(f"Applied log transformation to {col} (skewness: {skewness:.2f})")
            
            # Percentage change calculation
            if date_column != "None" and st.sidebar.checkbox("Calculate percentage changes"):
                for col in numeric_columns:
                    df[f"{col}_pct_change"] = df[col].pct_change() * 100
                    st.info(f"Added percentage change for {col}")
            
            # Moving averages
            if date_column != "None" and st.sidebar.checkbox("Calculate moving averages"):
                window = st.sidebar.slider("Moving average window", 2, 365, 30)
                for col in numeric_columns:
                    df[f"{col}_MA{window}"] = df[col].rolling(window=window).mean()
                    st.info(f"Added {window}-period moving average for {col}")
            
            # Add Download PDF button to sidebar
            st.sidebar.markdown("---")
            if st.sidebar.button("Download Analysis Report (PDF)"):
                try:
                    # Collect financial ratios if they exist
                    financial_ratios = {}
                    if 'current_ratio' in locals():
                        financial_ratios['Current Ratio'] = {
                            'value': current_ratio,
                            'interpretation': 'Measures ability to pay short-term obligations'
                        }
                    if 'quick_ratio' in locals():
                        financial_ratios['Quick Ratio'] = {
                            'value': quick_ratio,
                            'interpretation': 'Measures ability to meet short-term obligations with liquid assets'
                        }
                    if 'debt_to_equity' in locals():
                        financial_ratios['Debt to Equity'] = {
                            'value': debt_to_equity,
                            'interpretation': 'Shows proportion of debt and equity used to finance assets'
                        }
                    if 'gross_profit_margin' in locals():
                        financial_ratios['Gross Profit Margin'] = {
                            'value': gross_profit_margin,
                            'interpretation': 'Shows percentage of revenue retained after direct costs'
                        }
                    
                    # Generate PDF
                    pdf_buffer = generate_pdf_report(df, date_column, numeric_columns, financial_ratios)
                    
                    # Create download button
                    st.sidebar.download_button(
                        label="Click to Download PDF",
                        data=pdf_buffer,
                        file_name="financial_analysis_report.pdf",
                        mime="application/pdf"
                    )
                    st.sidebar.success("PDF report generated successfully!")
                except Exception as e:
                    st.sidebar.error(f"Error generating PDF: {str(e)}")
            
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
                
                # Add Financial Ratio Calculator at the end
                st.markdown('<h2 class="section-header">Financial Ratio Calculator</h2>', unsafe_allow_html=True)
                
                # Create columns for input
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Balance Sheet Items")
                    current_assets = st.number_input("Current Assets", min_value=0.0, value=0.0, step=1000.0, key="current_assets")
                    current_liabilities = st.number_input("Current Liabilities", min_value=0.0, value=0.0, step=1000.0, key="current_liabilities")
                    total_assets = st.number_input("Total Assets", min_value=0.0, value=0.0, step=1000.0, key="total_assets")
                    total_liabilities = st.number_input("Total Liabilities", min_value=0.0, value=0.0, step=1000.0, key="total_liabilities")
                    total_equity = st.number_input("Total Equity", min_value=0.0, value=0.0, step=1000.0, key="total_equity")
                    inventory = st.number_input("Inventory", min_value=0.0, value=0.0, step=1000.0, key="inventory")
                
                with col2:
                    st.subheader("Income Statement Items")
                    revenue = st.number_input("Revenue/Sales", min_value=0.0, value=0.0, step=1000.0, key="revenue")
                    cost_of_goods_sold = st.number_input("Cost of Goods Sold", min_value=0.0, value=0.0, step=1000.0, key="cogs")
                    gross_profit = st.number_input("Gross Profit", min_value=0.0, value=0.0, step=1000.0, key="gross_profit")
                    operating_expenses = st.number_input("Operating Expenses", min_value=0.0, value=0.0, step=1000.0, key="op_expenses")
                    net_income = st.number_input("Net Income", min_value=0.0, value=0.0, step=1000.0, key="net_income")
                    interest_expense = st.number_input("Interest Expense", min_value=0.0, value=0.0, step=1000.0, key="interest_expense")
                
                # Calculate ratios
                st.markdown('<h3 class="subsection-header">Financial Ratios</h3>', unsafe_allow_html=True)
                
                # Create columns for ratio display
                ratio_col1, ratio_col2 = st.columns(2)
                
                with ratio_col1:
                    st.subheader("Liquidity Ratios")
                    # Current Ratio
                    current_ratio = current_assets / current_liabilities if current_liabilities != 0 else 0
                    st.metric("Current Ratio", f"{current_ratio:.2f}", 
                             help="Measures a company's ability to pay short-term obligations. > 1 is healthy.")
                    
                    # Quick Ratio
                    quick_assets = current_assets - inventory
                    quick_ratio = quick_assets / current_liabilities if current_liabilities != 0 else 0
                    st.metric("Quick Ratio", f"{quick_ratio:.2f}",
                             help="Measures ability to meet short-term obligations with most liquid assets. > 1 is healthy.")
                    
                    # Cash Ratio
                    cash_ratio = (current_assets - inventory) / current_liabilities if current_liabilities != 0 else 0
                    st.metric("Cash Ratio", f"{cash_ratio:.2f}",
                             help="Measures ability to pay short-term debt with cash and cash equivalents.")
                
                with ratio_col2:
                    st.subheader("Profitability Ratios")
                    # Gross Profit Margin
                    gross_profit_margin = (gross_profit / revenue * 100) if revenue != 0 else 0
                    st.metric("Gross Profit Margin", f"{gross_profit_margin:.2f}%",
                             help="Shows percentage of revenue retained after direct costs.")
                    
                    # Operating Margin
                    operating_margin = ((revenue - cost_of_goods_sold - operating_expenses) / revenue * 100) if revenue != 0 else 0
                    st.metric("Operating Margin", f"{operating_margin:.2f}%",
                             help="Shows percentage of revenue retained after operating expenses.")
                    
                    # Net Profit Margin
                    net_profit_margin = (net_income / revenue * 100) if revenue != 0 else 0
                    st.metric("Net Profit Margin", f"{net_profit_margin:.2f}%",
                             help="Shows percentage of revenue retained as net income.")
                
                # Additional ratios in new columns
                ratio_col3, ratio_col4 = st.columns(2)
                
                with ratio_col3:
                    st.subheader("Efficiency Ratios")
                    # Inventory Turnover
                    inventory_turnover = cost_of_goods_sold / inventory if inventory != 0 else 0
                    st.metric("Inventory Turnover", f"{inventory_turnover:.2f}",
                             help="Shows how many times inventory is sold and replaced over a period.")
                    
                    # Asset Turnover
                    asset_turnover = revenue / total_assets if total_assets != 0 else 0
                    st.metric("Asset Turnover", f"{asset_turnover:.2f}",
                             help="Shows how efficiently assets are used to generate revenue.")
                
                with ratio_col4:
                    st.subheader("Leverage Ratios")
                    # Debt to Equity Ratio
                    debt_to_equity = total_liabilities / total_equity if total_equity != 0 else 0
                    st.metric("Debt to Equity Ratio", f"{debt_to_equity:.2f}",
                             help="Shows proportion of debt and equity used to finance assets.")
                    
                    # Interest Coverage Ratio
                    interest_coverage = (net_income + interest_expense) / interest_expense if interest_expense != 0 else 0
                    st.metric("Interest Coverage Ratio", f"{interest_coverage:.2f}",
                             help="Shows ability to pay interest on outstanding debt.")
                
                # Add ratio interpretation
                st.markdown('<h3 class="subsection-header">Ratio Interpretation</h3>', unsafe_allow_html=True)
                st.markdown("""
                - **Current Ratio**: > 1.5 is healthy, indicating good short-term financial health
                - **Quick Ratio**: > 1 is good, showing strong liquidity without relying on inventory
                - **Gross Profit Margin**: Varies by industry, but higher is generally better
                - **Operating Margin**: Shows operational efficiency, higher is better
                - **Net Profit Margin**: Indicates overall profitability, varies by industry
                - **Inventory Turnover**: Higher is better, showing efficient inventory management
                - **Debt to Equity**: < 2 is generally considered healthy
                - **Interest Coverage**: > 2 is healthy, showing good ability to service debt
                """)
            
            with tab2:
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
                            st.markdown(f'<h3 class="subsection-header">Analysis for {col}</h3>', unsafe_allow_html=True)
                            
                            # Create two columns for the first row of graphs
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Time series plot
                                fig = px.line(df, x=date_column, y=col, title=f"{col} Over Time")
                                fig.update_layout(
                                    title_x=0.5,
                                    title_font_size=20,
                                    title_font_color='#2C3E50'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                # Moving averages
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
                                fig.update_layout(
                                    title_x=0.5,
                                    title_font_size=20,
                                    title_font_color='#2C3E50'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Seasonal decomposition if enough data
                            if len(df) >= 14:
                                try:
                                    st.markdown(f'<h3 class="subsection-header">Seasonal Decomposition for {col}</h3>', unsafe_allow_html=True)
                                    
                                    # Create two columns for decomposition plots
                                    dec_col1, dec_col2 = st.columns(2)
                                    
                                    decomposition = seasonal_decompose(df[col], model='additive', period=7)
                                    
                                    with dec_col1:
                                        # First two plots (Observed and Trend)
                                        fig1 = go.Figure()
                                        fig1.add_trace(go.Scatter(x=df[date_column], y=decomposition.observed, 
                                                                 mode='lines', name='Observed'))
                                        fig1.update_layout(
                                            title='Observed',
                                            title_x=0.5,
                                            title_font_size=18,
                                            title_font_color='#2C3E50',
                                            height=300
                                        )
                                        st.plotly_chart(fig1, use_container_width=True)
                                        
                                        fig2 = go.Figure()
                                        fig2.add_trace(go.Scatter(x=df[date_column], y=decomposition.trend, 
                                                                 mode='lines', name='Trend'))
                                        fig2.update_layout(
                                            title='Trend',
                                            title_x=0.5,
                                            title_font_size=18,
                                            title_font_color='#2C3E50',
                                            height=300
                                        )
                                        st.plotly_chart(fig2, use_container_width=True)
                                    
                                    with dec_col2:
                                        # Last two plots (Seasonal and Residual)
                                        fig3 = go.Figure()
                                        fig3.add_trace(go.Scatter(x=df[date_column], y=decomposition.seasonal, 
                                                                 mode='lines', name='Seasonal'))
                                        fig3.update_layout(
                                            title='Seasonal',
                                            title_x=0.5,
                                            title_font_size=18,
                                            title_font_color='#2C3E50',
                                            height=300
                                        )
                                        st.plotly_chart(fig3, use_container_width=True)
                                        
                                        fig4 = go.Figure()
                                        fig4.add_trace(go.Scatter(x=df[date_column], y=decomposition.resid, 
                                                                 mode='lines', name='Residual'))
                                        fig4.update_layout(
                                            title='Residual',
                                            title_x=0.5,
                                            title_font_size=18,
                                            title_font_color='#2C3E50',
                                            height=300
                                        )
                                        st.plotly_chart(fig4, use_container_width=True)
                                        
                                except Exception as e:
                                    st.warning(f"Could not perform seasonal decomposition: {e}")
                
                # Scatter plot
                st.markdown('<h2 class="section-header">Scatter Plot Analysis</h2>', unsafe_allow_html=True)
                if len(num_cols) >= 2:
                    x_col = st.selectbox("Select X-axis column", options=num_cols)
                    y_col = st.selectbox("Select Y-axis column", options=[col for col in num_cols if col != x_col])
                    
                    fig = px.scatter(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
                    fig.update_layout(
                        title_x=0.5,
                        title_font_size=20,
                        title_font_color='#2C3E50'
                    )
                    st.plotly_chart(fig)
                
                # Category visualization
                cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                if cat_cols:
                    st.markdown('<h2 class="section-header">Categorical Data Analysis</h2>', unsafe_allow_html=True)
                    cat_col = st.selectbox("Select categorical column", options=cat_cols)
                    
                    # Count plot
                    value_counts = df[cat_col].value_counts().reset_index()
                    value_counts.columns = ['Category', 'count']
                    fig = px.bar(value_counts, x='Category', y='count', 
                                 title=f"Count of {cat_col}")
                    fig.update_layout(
                        title_x=0.5,
                        title_font_size=20,
                        title_font_color='#2C3E50'
                    )
                    st.plotly_chart(fig)
                    
                    # Relation with numerical variable
                    if num_cols:
                        num_col = st.selectbox("Select numerical column for analysis with categories", 
                                              options=num_cols)
                        
                        fig = px.box(df, x=cat_col, y=num_col, 
                                     title=f"{num_col} Distribution by {cat_col}")
                        fig.update_layout(
                            title_x=0.5,
                            title_font_size=20,
                            title_font_color='#2C3E50'
                        )
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
                            "Polynomial Regression": Pipeline([
                                ('poly', PolynomialFeatures(degree=2)),
                                ('linear', LinearRegression())
                            ]),
                            "Random Forest": RandomForestRegressor(random_state=random_state),
                            "Gradient Boosting": GradientBoostingRegressor(random_state=random_state),
                            "XGBoost": xgb.XGBRegressor(random_state=random_state),
                            "AdaBoost": AdaBoostRegressor(random_state=random_state),
                            "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=5)
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
                st.markdown('<h1 class="main-header">Future Predictions</h1>', unsafe_allow_html=True)
                
                if date_column != "None":
                    # Check if a model has been trained
                    if 'best_model' in st.session_state:
                        # Create tabs for different prediction types
                        pred_tab1, pred_tab2 = st.tabs(["Time Series Forecasting", "Normal Predictions"])
                        
                        with pred_tab1:
                            st.markdown('<h2 class="section-header">Time Series Forecasting</h2>', unsafe_allow_html=True)
                            
                            # Create columns for model selection and parameters
                            model_col1, model_col2 = st.columns(2)
                            
                            with model_col1:
                                selected_model = st.selectbox(
                                    "Select Time Series Model",
                                    options=["ARIMA", "Exponential Smoothing", "SARIMA", "Machine Learning Model"],
                                    key="ts_model_select"
                                )
                            
                            with model_col2:
                                forecast_periods = st.slider(
                                    "Forecast Periods",
                                    min_value=1,
                                    max_value=365,
                                    value=30,
                                    key="forecast_periods"
                                )
                            
                            # Select target column for forecasting
                            target_col = st.selectbox(
                                "Select column to forecast",
                                options=num_cols,
                                key="forecast_target"
                            )
                            
                            if st.button("Generate Time Series Forecast"):
                                try:
                                    # Prepare the time series data
                                    ts_data = df.set_index(date_column)[target_col]
                                    
                                    # Check for stationarity
                                    adf_result = adfuller(ts_data)
                                    st.info(f"ADF Test p-value: {adf_result[1]:.4f}")
                                    
                                    if selected_model == "ARIMA":
                                        st.markdown('<h3 class="subsection-header">ARIMA Model</h3>', unsafe_allow_html=True)
                                        
                                        # Model parameters
                                        p = st.slider("AR order (p)", 0, 5, 1, key="p")
                                        d = st.slider("Difference order (d)", 0, 2, 1, key="d")
                                        q = st.slider("MA order (q)", 0, 5, 1, key="q")
                                        
                                        try:
                                            # Check for stationarity
                                            adf_result = adfuller(ts_data)
                                            st.info(f"ADF Test p-value: {adf_result[1]:.4f}")
                                            
                                            # Fit ARIMA model
                                            model = ARIMA(ts_data, order=(p, d, q))
                                            model_fit = model.fit()
                                            
                                            # Get the fitted values for historical data
                                            fitted_values = model_fit.fittedvalues
                                            
                                            # Make forecast with prediction intervals
                                            forecast_result = model_fit.get_forecast(steps=forecast_periods)
                                            forecast = forecast_result.predicted_mean
                                            forecast_ci = forecast_result.conf_int(alpha=0.05)
                                            
                                            # Generate future dates
                                            future_dates = pd.date_range(start=ts_data.index[-1], periods=forecast_periods+1)[1:]
                                            
                                            # Plot results
                                            fig = go.Figure()
                                            
                                            # Add historical data
                                            fig.add_trace(go.Scatter(
                                                x=ts_data.index,
                                                y=ts_data.values,
                                                mode='lines',
                                                name='Historical',
                                                line=dict(color='blue')
                                            ))
                                            
                                            # Add fitted values
                                            fig.add_trace(go.Scatter(
                                                x=ts_data.index,
                                                y=fitted_values,
                                                mode='lines',
                                                name='Fitted',
                                                line=dict(color='green', dash='dash')
                                            ))
                                            
                                            # Add forecast
                                            fig.add_trace(go.Scatter(
                                                x=future_dates,
                                                y=forecast,
                                                mode='lines',
                                                name='Forecast',
                                                line=dict(color='red')
                                            ))
                                            
                                            # Add confidence intervals
                                            fig.add_trace(go.Scatter(
                                                x=future_dates,
                                                y=forecast_ci.iloc[:, 1],
                                                mode='lines',
                                                name='Upper Bound',
                                                line=dict(width=0),
                                                showlegend=False
                                            ))
                                            
                                            fig.add_trace(go.Scatter(
                                                x=future_dates,
                                                y=forecast_ci.iloc[:, 0],
                                                mode='lines',
                                                name='Lower Bound',
                                                line=dict(width=0),
                                                fill='tonexty',
                                                fillcolor='rgba(255, 0, 0, 0.2)',
                                                showlegend=False
                                            ))
                                            
                                            fig.update_layout(
                                                title=f'ARIMA({p},{d},{q}) Forecast',
                                                xaxis_title='Date',
                                                yaxis_title='Value',
                                                title_x=0.5,
                                                title_font_size=20,
                                                title_font_color='#2C3E50',
                                                hovermode='x unified'
                                            )
                                            
                                            st.plotly_chart(fig, use_container_width=True)
                                            
                                            # Display model metrics
                                            st.markdown('<h4 class="plot-title">Model Metrics</h4>', unsafe_allow_html=True)
                                            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                                            
                                            with metrics_col1:
                                                st.metric("AIC", f"{model_fit.aic:.2f}")
                                            with metrics_col2:
                                                st.metric("BIC", f"{model_fit.bic:.2f}")
                                            with metrics_col3:
                                                st.metric("HQIC", f"{model_fit.hqic:.2f}")
                                            
                                            # Display forecast data
                                            st.markdown('<h4 class="plot-title">Forecast Data</h4>', unsafe_allow_html=True)
                                            forecast_df = pd.DataFrame({
                                                'Date': future_dates,
                                                'Forecast': forecast,
                                                'Lower Bound': forecast_ci.iloc[:, 0],
                                                'Upper Bound': forecast_ci.iloc[:, 1]
                                            })
                                            st.dataframe(forecast_df)
                                            
                                            # Display model summary
                                            st.markdown('<h4 class="plot-title">Model Summary</h4>', unsafe_allow_html=True)
                                            st.text(model_fit.summary().tables[0].as_text())
                                            
                                        except Exception as e:
                                            st.error(f"Error in ARIMA forecasting: {str(e)}")
                                            st.info("Try adjusting the model parameters (p, d, q) to better fit your data.")
                                    
                                    elif selected_model == "Exponential Smoothing":
                                        st.markdown('<h3 class="subsection-header">Exponential Smoothing</h3>', unsafe_allow_html=True)
                                        
                                        # Model parameters
                                        trend_type = st.selectbox(
                                            "Trend Type",
                                            options=['add', 'mul', None],
                                            key="trend"
                                        )
                                        seasonal_type = st.selectbox(
                                            "Seasonal Type",
                                            options=['add', 'mul', None],
                                            key="seasonal"
                                        )
                                        seasonal_periods = st.slider(
                                            "Seasonal Periods",
                                            min_value=2,
                                            max_value=12,
                                            value=7,
                                            key="seasonal_periods"
                                        )
                                        
                                        # Fit model
                                        model = ExponentialSmoothing(
                                            ts_data,
                                            trend=trend_type,
                                            seasonal=seasonal_type,
                                            seasonal_periods=seasonal_periods
                                        ).fit()
                                        
                                        # Make forecast with prediction intervals
                                        forecast_result = model.get_prediction(start=len(ts_data), end=len(ts_data) + forecast_periods - 1)
                                        forecast = forecast_result.predicted_mean
                                        forecast_ci = forecast_result.conf_int(alpha=0.05)
                                        future_dates = pd.date_range(start=ts_data.index[-1], periods=forecast_periods+1)[1:]
                                        
                                        # Plot results
                                        fig = go.Figure()
                                        fig.add_trace(go.Scatter(
                                            x=ts_data.index,
                                            y=ts_data.values,
                                            mode='lines',
                                            name='Historical',
                                            line=dict(color='blue')
                                        ))
                                        fig.add_trace(go.Scatter(
                                            x=future_dates,
                                            y=forecast,
                                            mode='lines',
                                            name='Forecast',
                                            line=dict(color='red')
                                        ))
                                        fig.add_trace(go.Scatter(
                                            x=future_dates,
                                            y=forecast_ci.iloc[:, 1],
                                            mode='lines',
                                            name='Upper Bound',
                                            line=dict(width=0),
                                            showlegend=False
                                        ))
                                        fig.add_trace(go.Scatter(
                                            x=future_dates,
                                            y=forecast_ci.iloc[:, 0],
                                            mode='lines',
                                            name='Lower Bound',
                                            line=dict(width=0),
                                            fill='tonexty',
                                            fillcolor='rgba(255, 0, 0, 0.2)',
                                            showlegend=False
                                        ))
                                        fig.update_layout(
                                            title='Exponential Smoothing Forecast',
                                            title_x=0.5,
                                            title_font_size=20,
                                            title_font_color='#2C3E50'
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                        
                                    elif selected_model == "SARIMA":
                                        st.markdown('<h3 class="subsection-header">SARIMA Model</h3>', unsafe_allow_html=True)
                                        
                                        # Model parameters
                                        p = st.slider("AR order (p)", 0, 3, 1, key="sarima_p")
                                        d = st.slider("Difference order (d)", 0, 2, 1, key="sarima_d")
                                        q = st.slider("MA order (q)", 0, 3, 1, key="sarima_q")
                                        P = st.slider("Seasonal AR order (P)", 0, 2, 1, key="sarima_P")
                                        D = st.slider("Seasonal difference order (D)", 0, 2, 1, key="sarima_D")
                                        Q = st.slider("Seasonal MA order (Q)", 0, 2, 1, key="sarima_Q")
                                        m = st.slider("Seasonal period (m)", 2, 12, 7, key="sarima_m")
                                        
                                        # Fit SARIMA model
                                        model = SARIMAX(
                                            ts_data,
                                            order=(p, d, q),
                                            seasonal_order=(P, D, Q, m)
                                        )
                                        model_fit = model.fit(disp=False)
                                        
                                        # Make forecast with prediction intervals
                                        forecast_result = model_fit.get_forecast(steps=forecast_periods)
                                        forecast = forecast_result.predicted_mean
                                        forecast_ci = forecast_result.conf_int(alpha=0.05)
                                        future_dates = pd.date_range(start=ts_data.index[-1], periods=forecast_periods+1)[1:]
                                        
                                        # Plot results
                                        fig = go.Figure()
                                        fig.add_trace(go.Scatter(
                                            x=ts_data.index,
                                            y=ts_data.values,
                                            mode='lines',
                                            name='Historical',
                                            line=dict(color='blue')
                                        ))
                                        fig.add_trace(go.Scatter(
                                            x=future_dates,
                                            y=forecast,
                                            mode='lines',
                                            name='Forecast',
                                            line=dict(color='red')
                                        ))
                                        fig.add_trace(go.Scatter(
                                            x=future_dates,
                                            y=forecast_ci.iloc[:, 1],
                                            mode='lines',
                                            name='Upper Bound',
                                            line=dict(width=0),
                                            showlegend=False
                                        ))
                                        fig.add_trace(go.Scatter(
                                            x=future_dates,
                                            y=forecast_ci.iloc[:, 0],
                                            mode='lines',
                                            name='Lower Bound',
                                            line=dict(width=0),
                                            fill='tonexty',
                                            fillcolor='rgba(255, 0, 0, 0.2)',
                                            showlegend=False
                                        ))
                                        fig.update_layout(
                                            title=f'SARIMA({p},{d},{q})({P},{D},{Q},{m}) Forecast',
                                            title_x=0.5,
                                            title_font_size=20,
                                            title_font_color='#2C3E50'
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                        
                                    elif selected_model == "Machine Learning Model":
                                        st.markdown('<h3 class="subsection-header">Machine Learning Model Forecast</h3>', unsafe_allow_html=True)
                                        
                                        # Use the best trained model from the Modeling tab
                                        best_model = st.session_state['best_model']
                                        best_model_name = st.session_state['best_model_name']
                                        feature_columns = st.session_state['feature_columns']
                                        scaler = st.session_state['scaler']
                                        
                                        # Generate future dates
                                        last_date = df[date_column].max()
                                        future_dates = pd.date_range(start=last_date, periods=forecast_periods+1)[1:]
                                        
                                        # Create future dataframe
                                        future_df = pd.DataFrame({date_column: future_dates})
                                        
                                        # Generate future features using trend analysis and seasonality
                                        for feature in feature_columns:
                                            if feature in df.columns:
                                                historical_values = df[feature].values
                                                
                                                # Fit trend
                                                X_trend = np.arange(len(historical_values)).reshape(-1, 1)
                                                trend_model = LinearRegression()
                                                trend_model.fit(X_trend, historical_values)
                                                
                                                # Generate trend values
                                                X_future_trend = np.arange(len(historical_values), 
                                                                          len(historical_values) + forecast_periods).reshape(-1, 1)
                                                trend_values = trend_model.predict(X_future_trend)
                                                
                                                # Add seasonality if enough data points
                                                if len(historical_values) >= 14:  # At least 2 weeks of data
                                                    # Calculate seasonal pattern
                                                    seasonal_pattern = np.zeros(7)  # Weekly pattern
                                                    for i in range(7):
                                                        mask = np.arange(len(historical_values)) % 7 == i
                                                        if np.sum(mask) > 0:
                                                            seasonal_pattern[i] = np.mean(historical_values[mask])
                                                    
                                                    # Apply seasonality
                                                    seasonal_values = np.tile(seasonal_pattern, forecast_periods // 7 + 1)[:forecast_periods]
                                                    trend_values += seasonal_values - np.mean(seasonal_pattern)
                                                
                                                # Add random noise based on historical volatility
                                                historical_std = np.std(historical_values)
                                                random_noise = np.random.normal(0, historical_std * 0.1, forecast_periods)
                                                future_df[feature] = trend_values + random_noise
                                        
                                        # Scale features and make predictions
                                        X_future = future_df[feature_columns]
                                        X_future_scaled = scaler.transform(X_future)
                                        forecast = best_model.predict(X_future_scaled)
                                        
                                        # Calculate prediction intervals
                                        y_pred_historical = best_model.predict(scaler.transform(df[feature_columns]))
                                        prediction_errors = df[target_col] - y_pred_historical
                                        error_std = np.std(prediction_errors)
                                        
                                        # Add confidence intervals
                                        forecast_lower = forecast - 1.96 * error_std
                                        forecast_upper = forecast + 1.96 * error_std
                                        
                                        # Plot results
                                        fig = go.Figure()
                                        fig.add_trace(go.Scatter(
                                            x=df[date_column],
                                            y=df[target_col],
                                            mode='lines',
                                            name='Historical',
                                            line=dict(color='blue')
                                        ))
                                        fig.add_trace(go.Scatter(
                                            x=future_dates,
                                            y=forecast,
                                            mode='lines',
                                            name='Forecast',
                                            line=dict(color='red')
                                        ))
                                        fig.add_trace(go.Scatter(
                                            x=future_dates,
                                            y=forecast_upper,
                                            mode='lines',
                                            name='Upper Bound',
                                            line=dict(width=0),
                                            showlegend=False
                                        ))
                                        fig.add_trace(go.Scatter(
                                            x=future_dates,
                                            y=forecast_lower,
                                            mode='lines',
                                            name='Lower Bound',
                                            line=dict(width=0),
                                            fill='tonexty',
                                            fillcolor='rgba(255, 0, 0, 0.2)',
                                            showlegend=False
                                        ))
                                        fig.update_layout(
                                            title=f'{best_model_name} Forecast',
                                            title_x=0.5,
                                            title_font_size=20,
                                            title_font_color='#2C3E50'
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Display model metrics for time series models
                                    if selected_model != "Machine Learning Model":
                                        st.markdown('<h4 class="plot-title">Model Metrics</h4>', unsafe_allow_html=True)
                                        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                                        
                                        with metrics_col1:
                                            st.metric("AIC", f"{model_fit.aic:.2f}")
                                        with metrics_col2:
                                            st.metric("BIC", f"{model_fit.bic:.2f}")
                                        with metrics_col3:
                                            st.metric("HQIC", f"{model_fit.hqic:.2f}")
                                    
                                    # Display forecast data
                                    st.markdown('<h4 class="plot-title">Forecast Data</h4>', unsafe_allow_html=True)
                                    if selected_model == "Machine Learning Model":
                                        forecast_df = pd.DataFrame({
                                            'Date': future_dates,
                                            'Forecast': forecast,
                                            'Lower Bound': forecast_lower,
                                            'Upper Bound': forecast_upper
                                        })
                                    else:
                                        forecast_df = pd.DataFrame({
                                            'Date': future_dates,
                                            'Forecast': forecast,
                                            'Lower Bound': forecast_ci.iloc[:, 0],
                                            'Upper Bound': forecast_ci.iloc[:, 1]
                                        })
                                    st.dataframe(forecast_df)
                                    
                                except Exception as e:
                                    st.error(f"Error in forecasting: {e}")
                        
                        with pred_tab2:
                            st.markdown('<h2 class="section-header">Normal Predictions</h2>', unsafe_allow_html=True)
                            
                            # Get the trained model information
                            best_model = st.session_state['best_model']
                            best_model_name = st.session_state['best_model_name']
                            feature_columns = st.session_state['feature_columns']
                            scaler = st.session_state['scaler']
                            target_column = st.session_state['target_column']
                            
                            st.info(f"Using the best model from training: {best_model_name}")
                            
                            # Forecast settings
                            forecast_periods = st.slider(
                                "Number of periods to forecast",
                                min_value=1,
                                max_value=365,
                                value=30,
                                key="normal_forecast_periods"
                            )
                            
                            if st.button("Generate Normal Forecast"):
                                try:
                                    # Generate future dates
                                    last_date = df[date_column].max()
                                    future_dates = pd.date_range(start=last_date, periods=forecast_periods+1)[1:]
                                    
                                    # Create future dataframe
                                    future_df = pd.DataFrame({date_column: future_dates})
                                    
                                    # Generate future features using trend analysis
                                    for feature in feature_columns:
                                        if feature in df.columns:
                                            historical_values = df[feature].values
                                            X_trend = np.arange(len(historical_values)).reshape(-1, 1)
                                            trend_model = LinearRegression()
                                            trend_model.fit(X_trend, historical_values)
                                            X_future_trend = np.arange(len(historical_values), 
                                                                      len(historical_values) + forecast_periods).reshape(-1, 1)
                                            trend_values = trend_model.predict(X_future_trend)
                                            historical_std = np.std(historical_values)
                                            random_noise = np.random.normal(0, historical_std * 0.1, forecast_periods)
                                            future_df[feature] = trend_values + random_noise
                                    
                                    # Scale features and make predictions
                                    X_future = future_df[feature_columns]
                                    X_future_scaled = scaler.transform(X_future)
                                    forecast = best_model.predict(X_future_scaled)
                                    
                                    # Calculate prediction intervals
                                    y_pred_historical = best_model.predict(scaler.transform(df[feature_columns]))
                                    prediction_errors = df[target_column] - y_pred_historical
                                    error_std = np.std(prediction_errors)
                                    
                                    # Add confidence intervals
                                    forecast_lower = forecast - 1.96 * error_std
                                    forecast_upper = forecast + 1.96 * error_std
                                    
                                    # Plot results
                                    fig = go.Figure()
                                    
                                    # Add historical data
                                    fig.add_trace(go.Scatter(
                                        x=df[date_column],
                                        y=df[target_column],
                                        mode='lines',
                                        name='Historical',
                                        line=dict(color='blue')
                                    ))
                                    
                                    # Add forecast
                                    fig.add_trace(go.Scatter(
                                        x=future_dates,
                                        y=forecast,
                                        mode='lines',
                                        name='Forecast',
                                        line=dict(color='red')
                                    ))
                                    
                                    # Add confidence intervals
                                    fig.add_trace(go.Scatter(
                                        x=future_dates,
                                        y=forecast_upper,
                                        mode='lines',
                                        name='Upper Bound',
                                        line=dict(width=0),
                                        showlegend=False
                                    ))
                                    
                                    fig.add_trace(go.Scatter(
                                        x=future_dates,
                                        y=forecast_lower,
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
                                        hovermode='x unified',
                                        showlegend=True,
                                        title_x=0.5,
                                        title_font_size=20,
                                        title_font_color='#2C3E50'
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Display forecast data with confidence intervals
                                    st.markdown('<h4 class="plot-title">Forecast Data</h4>', unsafe_allow_html=True)
                                    forecast_df = pd.DataFrame({
                                        'Date': future_dates,
                                        'Forecast': forecast,
                                        'Lower Bound': forecast_lower,
                                        'Upper Bound': forecast_upper
                                    })
                                    st.dataframe(forecast_df)
                                    
                                    # Show feature importance if available
                                    if hasattr(best_model, 'feature_importances_'):
                                        st.markdown('<h4 class="plot-title">Feature Importance</h4>', unsafe_allow_html=True)
                                        importance_df = pd.DataFrame({
                                            'Feature': feature_columns,
                                            'Importance': best_model.feature_importances_
                                        }).sort_values('Importance', ascending=False)
                                        
                                        fig = px.bar(
                                            importance_df,
                                            x='Feature',
                                            y='Importance',
                                            title='Feature Importance'
                                        )
                                        fig.update_layout(
                                            title_x=0.5,
                                            title_font_size=20,
                                            title_font_color='#2C3E50'
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                    
                                except Exception as e:
                                    st.error(f"Error in prediction: {e}")
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