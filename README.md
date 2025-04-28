# Financial AutoML Application

A comprehensive financial analysis and machine learning application built with Streamlit that provides automated analysis, visualization, and forecasting capabilities for financial data.

## Features

### Data Input and Preprocessing
- Support for multiple file formats (CSV, Excel)
- Automatic date column detection
- Flexible numeric column selection
- Advanced data cleaning options:
  - Multiple missing value handling strategies
  - Duplicate removal
  - Outlier detection and handling
  - Advanced data transformations (log, percentage change, moving averages)

### Data Analysis
- Comprehensive summary statistics
- Data type analysis
- Correlation analysis with heatmap visualization
- Distribution analysis for numerical columns
- Financial ratio calculator with:
  - Liquidity ratios
  - Profitability ratios
  - Efficiency ratios
  - Leverage ratios
  - Industry benchmarks and interpretations

### Data Visualization
- Time series analysis with multiple components:
  - Trend analysis
  - Seasonal decomposition
  - Advanced time series models (ARIMA, SARIMA, Exponential Smoothing)
- Interactive plots with hover information
- Side-by-side visualization options
- Customizable plot parameters

### Machine Learning Models
- Classification models:
  - Logistic Regression
  - Random Forest
  - Support Vector Machine
  - Gradient Boosting
  - Decision Tree
  - K-Nearest Neighbors
- Regression models:
  - Linear Regression
  - Multilinear Regression
  - Polynomial Regression
  - AdaBoost
  - K-Nearest Neighbors
- Model evaluation metrics
- Hyperparameter tuning
- Cross-validation

### Future Predictions
- Advanced forecasting capabilities:
  - ARIMA modeling with parameter selection
  - SARIMA modeling for seasonal data
  - Exponential Smoothing with trend and seasonal components
- Confidence intervals
- Model diagnostics (AIC, BIC, HQIC)
- Interactive forecast visualization

### Additional Features
- PDF report generation
- Interactive parameter selection
- Real-time model updates
- Comprehensive error handling
- User-friendly interface

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/financial-automl.git
cd financial-automl
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
streamlit run app.py
```

2. Upload your financial data file (CSV or Excel format)
3. Select your date and numeric columns
4. Choose your analysis options
5. Explore the various tabs for different analyses
6. Generate and download PDF reports

## Requirements

- Python 3.7+
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Statsmodels
- ReportLab
- Openpyxl (for Excel support)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 