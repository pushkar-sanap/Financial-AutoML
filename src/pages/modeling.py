import streamlit as st
import pandas as pd
import numpy as np
from src.utils.data_processing import prepare_data_for_modeling
from src.models.ml_models import get_available_models, train_and_evaluate_models, find_best_model
from src.visualization.data_viz import plot_model_performance, plot_actual_vs_predicted, plot_feature_importance

def show_modeling_page(df):
    """Show modeling page content"""
    st.header("Predictive Modeling")
    
    # Get numerical columns
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    
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
            X, y, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = prepare_data_for_modeling(
                df, target_col, selected_features, test_size, random_state
            )
            
            # Get available models
            models = get_available_models(random_state)
            
            # Model selection
            selected_models = st.multiselect(
                "Select models to train",
                options=list(models.keys()),
                default=["Linear Regression", "Random Forest"]
            )
            
            if selected_models and st.button("Train Models"):
                st.subheader("Model Performance")
                
                # Train and evaluate models
                results_df, trained_models = train_and_evaluate_models(
                    X_train_scaled, X_test_scaled, y_train, y_test, selected_models, models
                )
                
                # Display results
                st.dataframe(results_df)
                
                # Plot model performance comparison
                plot_model_performance(results_df)
                
                # Find best model
                best_model_name, best_r2 = find_best_model(results_df)
                st.success(f"Best model: {best_model_name} with R2 Score: {best_r2:.4f}")
                
                # Get best model
                best_model = trained_models[best_model_name]
                
                # Plot actual vs predicted
                y_pred_best = best_model.predict(X_test_scaled)
                plot_actual_vs_predicted(y_test, y_pred_best, best_model_name)
                
                # Feature importance for tree-based models
                if best_model_name in ["Random Forest", "Gradient Boosting", "XGBoost"]:
                    st.subheader("Feature Importance")
                    plot_feature_importance(best_model, X.columns, best_model_name)
                
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