import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, QuantileTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                            roc_curve, roc_auc_score, accuracy_score,
                            precision_score, recall_score, f1_score)

# Disable the PyplotGlobalUseWarning
st.set_option('deprecation.showPyplotGlobalUse', False)

def show_fraud_analysis_page(df=None):
    st.header("Fraud Detection Analysis")
    
    st.markdown("""
    This section helps identify potential fraudulent transactions using machine learning.
    It compares SVM and KNN models for fraud detection using the transaction_data.csv file.
    """)
    
    # Load data from specific file path
    file_path = r'src/assets/transaction_data.csv'
    try:
        data = pd.read_csv(file_path)
        
        # Check if first column has all data merged
        if data.columns.size == 1:
            data = pd.read_csv(file_path, sep=',')
            data = data[data.columns[0]].str.split(',', expand=True)
            data.columns = ['TransactionID', 'Timestamp', 'Amount', 'TransactionType', 
                          'Location', 'IPAddress', 'UserID', 'IsFraud']
        
        with st.expander("View Transaction Data"):
            st.dataframe(data.head())
            st.write(f"Dataset shape: {data.shape}")
            
        if 'IsFraud' not in data.columns:
            st.error("The transaction_data.csv file doesn't contain an 'IsFraud' column.")
            st.write("Available columns:", data.columns.tolist())
            return
        
        # Data Preprocessing
        target_column = 'IsFraud'
        X = data.drop(target_column, axis=1)
        y = data[target_column].astype(int)

        numerical_features = ['Amount']
        categorical_features = ['TransactionType', 'Location']
        
        numerical_transformer = Pipeline([
            ('scaler', StandardScaler()),
            ('quantile', QuantileTransformer(output_distribution='normal', n_quantiles=100))
        ])

        categorical_transformer = OneHotEncoder(handle_unknown='ignore')

        preprocessor = ColumnTransformer(transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        # Model Setup
        with st.spinner("Training models..."):
            svm_pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', SVC(probability=True, random_state=42))
            ])

            knn_pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', KNeighborsClassifier())
            ])

            svm_param_grid = {
                'classifier__C': [0.1, 1, 10],
                'classifier__kernel': ['linear', 'rbf'],
                'classifier__gamma': ['scale', 'auto']
            }

            knn_param_grid = {
                'classifier__n_neighbors': [3, 5, 7],
                'classifier__weights': ['uniform', 'distance'],
                'classifier__metric': ['euclidean', 'manhattan']
            }

            svm_grid = GridSearchCV(svm_pipeline, svm_param_grid, cv=3, scoring='f1', n_jobs=-1)
            knn_grid = GridSearchCV(knn_pipeline, knn_param_grid, cv=3, scoring='f1', n_jobs=-1)

            svm_grid.fit(X_train, y_train)
            knn_grid.fit(X_train, y_train)

            best_svm = svm_grid.best_estimator_
            best_knn = knn_grid.best_estimator_

        # Prediction & Evaluation
        y_pred_svm = best_svm.predict(X_test)
        y_pred_knn = best_knn.predict(X_test)

        y_proba_svm = best_svm.predict_proba(X_test)[:, 1]
        y_proba_knn = best_knn.predict_proba(X_test)[:, 1]

        metrics = {
            'Accuracy': [accuracy_score(y_test, y_pred_svm), accuracy_score(y_test, y_pred_knn)],
            'Precision': [precision_score(y_test, y_pred_svm), precision_score(y_test, y_pred_knn)],
            'Recall': [recall_score(y_test, y_pred_svm), recall_score(y_test, y_pred_knn)],
            'F1-Score': [f1_score(y_test, y_pred_svm), f1_score(y_test, y_pred_knn)],
            'ROC-AUC': [roc_auc_score(y_test, y_proba_svm), roc_auc_score(y_test, y_proba_knn)]
        }

        comparison_df = pd.DataFrame(metrics, index=['SVM', 'KNN']).T
        st.write("### Model Comparison (SVM vs KNN)")
        st.dataframe(comparison_df.style.format("{:.4f}"))

        # Fraud Analysis Report
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Transactions", len(y))
            st.metric("Fraudulent Transactions", y.sum())
        with col2:
            st.metric("Fraud Rate", f"{y.mean()*100:.2f}%")
            st.metric("Test Set Size", len(y_test))

        # Visualizations
        st.subheader("Model Performance Visualizations")

        # Confusion Matrices
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        sns.heatmap(confusion_matrix(y_test, y_pred_svm),
                    annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title('SVM Confusion Matrix')
        sns.heatmap(confusion_matrix(y_test, y_pred_knn),
                    annot=True, fmt='d', cmap='Greens', ax=axes[1])
        axes[1].set_title('KNN Confusion Matrix')
        st.pyplot(fig)

        # ROC Curves
        fig, ax = plt.subplots(figsize=(8, 6))
        fpr_svm, tpr_svm, _ = roc_curve(y_test, y_proba_svm)
        fpr_knn, tpr_knn, _ = roc_curve(y_test, y_proba_knn)
        ax.plot(fpr_svm, tpr_svm, label=f'SVM (AUC = {roc_auc_score(y_test, y_proba_svm):.2f})')
        ax.plot(fpr_knn, tpr_knn, label=f'KNN (AUC = {roc_auc_score(y_test, y_proba_knn):.2f})')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve Comparison')
        ax.legend()
        st.pyplot(fig)

        # Best Parameters
        with st.expander("Show Best Model Parameters"):
            st.write("**SVM Best Parameters:**")
            st.json({
                "classifier__C": svm_grid.best_params_['classifier__C'],
                "classifier__gamma": svm_grid.best_params_['classifier__gamma'],
                "classifier__kernel": svm_grid.best_params_['classifier__kernel']
            })
            
            st.write("**KNN Best Parameters:**")
            st.json({
                "classifier__metric": knn_grid.best_params_['classifier__metric'],
                "classifier__n_neighbors": knn_grid.best_params_['classifier__n_neighbors'],
                "classifier__weights": knn_grid.best_params_['classifier__weights']
            })
            
    except FileNotFoundError:
        st.error(f"Transaction data file not found at: {file_path}")
    except Exception as e:
        st.error(f"An error occurred while loading the transaction data: {str(e)}")