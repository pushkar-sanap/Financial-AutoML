import pandas as pd
import numpy as np
import streamlit as st

def load_data(uploaded_file):
    """Load data from uploaded file"""
    try:
        df = pd.read_csv(uploaded_file)
        st.success("Data loaded successfully!")
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def convert_date_column(df, date_column):
    """Convert a column to datetime and sort by it"""
    if date_column != "None":
        try:
            df[date_column] = pd.to_datetime(df[date_column])
            df = df.sort_values(by=date_column)
            st.info(f"Converted '{date_column}' to datetime and sorted data.")
        except Exception as e:
            st.error(f"Error converting date column: {e}")
    return df

def handle_missing_values(df, strategy):
    """Handle missing values according to selected strategy"""
    if strategy == "Drop":
        df = df.dropna()
        st.info("Dropped rows with missing values")
    elif strategy == "Fill with mean":
        for col in df.select_dtypes(include=np.number).columns:
            df[col].fillna(df[col].mean(), inplace=True)
        st.info("Filled numerical missing values with mean")
    elif strategy == "Fill with median":
        for col in df.select_dtypes(include=np.number).columns:
            df[col].fillna(df[col].median(), inplace=True)
        st.info("Filled numerical missing values with median")
    elif strategy == "Fill with 0":
        df.fillna(0, inplace=True)
        st.info("Filled missing values with 0")
    return df

def get_column_types(df):
    """Get numerical and categorical columns from DataFrame"""
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    return num_cols, cat_cols

def display_missing_values(df):
    """Display missing values information"""
    if df.isna().sum().sum() > 0:
        st.subheader("Missing Values")
        missing_df = pd.DataFrame({
            'Column': df.columns,
            'Missing Values': df.isna().sum().values,
            'Percentage': (df.isna().sum().values / len(df)) * 100
        })
        st.dataframe(missing_df)
        return True
    return False

def create_sample_data():
    """Create sample data for display when no file is uploaded"""
    sample_data = {
        'Date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
        'Revenue': [1000, 1200, 950, 1100, 1300],
        'Expenses': [800, 750, 820, 780, 850],
        'Profit': [200, 450, 130, 320, 450],
        'Category': ['Product A', 'Product B', 'Product A', 'Product C', 'Product B']
    }
    return pd.DataFrame(sample_data)

def prepare_data_for_modeling(df, target_col, feature_cols, test_size=0.2, random_state=42):
    """Prepare data for modeling"""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    X = df[feature_cols]
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
    
    return X, y, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler 