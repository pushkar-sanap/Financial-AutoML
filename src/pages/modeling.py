import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb

def show_modeling(df, numeric_cols):
    st.markdown('<h1 class="main-header">Modeling</h1>', unsafe_allow_html=True)
    st.info("This section will be used for building and evaluating machine learning models.")
    # Add your modeling functionalities here in the future
    if numeric_cols:
        st.subheader("Select Target and Features")
        target_column = st.selectbox("Select the target variable", options=numeric_cols)
        feature_columns = st.multiselect("Select the feature variables", options=[col for col in numeric_cols if col != target_column])

        if target_column and feature_columns:
            st.subheader("Model Selection and Training")
            model_type = st.selectbox("Choose a regression model",
                                      ["Linear Regression", "Polynomial Regression", "Random Forest",
                                       "Gradient Boosting", "AdaBoost", "K-Nearest Neighbors", "XGBoost"])

            if st.button("Train Model"):
                X = df[feature_columns]
                y = df[target_column]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                if model_type == "Linear Regression":
                    model = LinearRegression()
                elif model_type == "Polynomial Regression":
                    degree = st.slider("Polynomial Degree", 2, 5, 2)
                    model = Pipeline([('poly', PolynomialFeatures(degree=degree)),
                                      ('scaler', StandardScaler()),
                                      ('linear', LinearRegression())])
                elif model_type == "Random Forest":
                    model = RandomForestRegressor(random_state=42)
                elif model_type == "Gradient Boosting":
                    model = GradientBoostingRegressor(random_state=42)
                elif model_type == "AdaBoost":
                    model = AdaBoostRegressor(random_state=42)
                elif model_type == "K-Nearest Neighbors":
                    n_neighbors = st.slider("Number of Neighbors", 1, 10, 5)
                    model = KNeighborsRegressor(n_neighbors=n_neighbors)
                elif model_type == "XGBoost":
                    model = xgb.XGBRegressor(random_state=42)

                with st.spinner(f"Training {model_type} model..."):
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)

                    st.subheader("Model Evaluation")
                    st.metric("Mean Squared Error", f"{mse:.2f}")
                    st.metric("R-squared", f"{r2:.2f}")
                    st.metric("Mean Absolute Error", f"{mae:.2f}")
    else:
        st.info("No numerical columns available for modeling.")
