# Fin-Alysis: Automated Financial Analysis & Forecasting Tool

Fin-Alysis is a Streamlit based application that automates stock level financial analysis, exploratory data visualization, feature engineering, and predictive modeling using ML techniques. It has been designed to assist retail investors, and analysts in making data driven financial decisions.


## Features

- **Data Upload**: Upload CSV files containing financial data
- **Data Preprocessing**: Handle missing values, convert date columns
- **Exploratory Data Analysis**: View summary statistics, correlations, and distributions
- **Data Visualization**: Time series analysis, trend visualization, categorical data analysis
- **Automated Machine Learning**: Train multiple regression models and compare their performance
- **Financial Forecasting**: Predict future financial trends using the best model
- **Time Series Analysis**: Perform seasonal decomposition and exponential smoothing

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:

```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL displayed in your terminal (typically http://localhost:8501)

3. Upload your financial CSV file using the file uploader in the sidebar

4. Navigate through the tabs to explore your data, visualize trends, and train prediction models

## Sample Data Format

Your dataset should include:

- A Date column for time series operations
- Numerical metrics such as Revenue, Expenses, Profit
- Optional categorical columns like Product, Department, Category

## Example

A sample dataset might look like:

| Date       | Revenue | Expenses | Profit | Category  |
|------------|---------|----------|--------|-----------|
| 2023-01-01 | 1000    | 800      | 200    | Product A |
| 2023-01-02 | 1200    | 750      | 450    | Product B |
| 2023-01-03 | 950     | 820      | 130    | Product A |
| 2023-01-04 | 1100    | 780      | 320    | Product C |
| 2023-01-05 | 1300    | 850      | 450    | Product B |

## Contributors
[Pushkar Sanap](https://github.com/pushkar-sanap)
[Pavit Kaur](https://github.com/pavit15)
[Amey Jawale](https://github.com/ameyjawale)
[Ritvik Jeeda](https://github.com/Apollo-RJ)
