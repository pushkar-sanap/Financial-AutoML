# Financial AutoML App - User Guide

This guide provides detailed instructions on how to use the Financial AutoML application.

## Getting Started

1. Start the application by running `streamlit run app.py` in your terminal
2. The application will open in your default web browser

## Uploading Data

1. On the left sidebar, click on the "Upload your financial CSV file" button
2. Select your financial data CSV file from your computer
3. If you don't have a financial dataset, you can use the provided `sample_financial_data.csv` file

## Data Preprocessing

After uploading your data, you can:

1. Select a date column (if your data includes dates) from the dropdown in the sidebar
2. Choose a method to handle missing values, if any:
   - **None**: Leave missing values as is
   - **Drop**: Remove rows with missing values
   - **Fill with mean**: Replace missing values with the mean of the column
   - **Fill with median**: Replace missing values with the median of the column
   - **Fill with 0**: Replace missing values with zeros

## Exploring Your Data

The app has four main tabs to analyze your data:

### Data Analysis Tab

This tab provides basic exploratory data analysis (EDA):

1. **Raw Data Preview**: View the first few rows of your dataset
2. **Summary Statistics**: See statistical measures of your numerical columns
3. **Data Types**: View the data type of each column
4. **Correlation Analysis**: See how your numerical variables correlate with each other
5. **Distributions**: View the distribution of selected numerical columns

### Data Visualization Tab

This tab offers advanced visualizations:

1. **Time Series Analysis**: If you selected a date column, view how numerical columns change over time
2. **Moving Averages**: See 7-day and 30-day moving averages for selected metrics
3. **Seasonal Decomposition**: Break down time series into trend, seasonal, and residual components
4. **Scatter Plot Analysis**: Explore relationships between pairs of numerical variables
5. **Categorical Data Analysis**: Analyze the distribution of categorical variables and their relationship with numerical variables

### Modeling Tab

In this tab, you can train machine learning models:

1. Select a target column to predict
2. Choose which features to include in the model
3. Set the test size (percentage of data to use for testing)
4. Select which models to train:
   - Linear Regression
   - Ridge Regression
   - Lasso Regression
   - Random Forest
   - Gradient Boosting
   - XGBoost
5. Click "Train Models" to train and evaluate the selected models
6. Review model performance metrics (MSE, RMSE, MAE, R2 Score)
7. View the actual vs. predicted plot for the best model
8. For tree-based models, examine feature importance

### Future Predictions Tab

This tab allows you to forecast future values:

1. Set the number of periods to forecast
2. Click "Generate Forecast" to predict future values
3. View the historical data and forecast on a time series plot
4. Examine detailed forecast data in a table
5. Alternatively, click "Run Time Series Analysis" to use Holt-Winters exponential smoothing for forecasting

## Tips for Getting the Best Results

1. **Data Quality**: Ensure your data is clean and properly formatted
2. **Date Format**: If using time series forecasting, make sure your date column is properly formatted
3. **Feature Selection**: Choose relevant features for your prediction model
4. **Model Selection**: Try multiple models to find the one that works best for your data
5. **Test Size**: A test size of 20% is typically a good starting point

## Interpreting Results

1. **R2 Score**: Higher is better, indicates how well the model explains the variance in the data
2. **RMSE/MAE**: Lower is better, indicates the average prediction error
3. **Feature Importance**: Indicates which features have the most impact on predictions
4. **Forecast Confidence Interval**: The shaded area in the forecast plot indicates uncertainty

## Exporting Results

Currently, the app does not support exporting results to files. You can take screenshots or manually copy the data for your records. 