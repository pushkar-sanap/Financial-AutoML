import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Binarize the continuous columns based on a threshold (mean or median)
def binarize_data(df):
    # Binarize all numeric columns (excluding 'Quarter')
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        threshold = df[col].median()  # You can use .mean() if preferred
        df[col] = (df[col] > threshold).astype(int)  # Convert to 1/0 based on threshold
    return df

# Read the financial data CSV
def load_financial_data():
    df = pd.read_csv('src/assets/association_data.csv')
    return df

# Perform association analysis
def run_association_analysis(df):
    binarized_df = binarize_data(df)  # Binarize the continuous columns
    frequent_itemsets = apriori(binarized_df.drop('Quarter', axis=1), min_support=0.2, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    rules = rules.sort_values(by='lift', ascending=False)
    return rules

# Visualize association results
def plot_association_graphs(rules):
    # Top rules table
    top_rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10)
    top_rules['antecedents'] = top_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    top_rules['consequents'] = top_rules['consequents'].apply(lambda x: ', '.join(list(x)))
    
    st.write("### Top 10 Association Rules")
    st.dataframe(top_rules)

    # Visualizing the support vs. confidence
    st.write("### Support vs. Confidence Plot")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=rules, x="support", y="confidence", size="lift", hue="lift", palette="viridis", sizes=(20, 200), alpha=0.7, ax=ax)
    ax.set_title("Support vs. Confidence with Lift as Size")
    ax.set_xlabel("Support")
    ax.set_ylabel("Confidence")
    ax.legend(title="Lift", bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig)

    # Visualizing the lift distribution
    st.write("### Distribution of Lift")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(rules['lift'], kde=True, color='skyblue', ax=ax)
    ax.set_title("Distribution of Lift")
    ax.set_xlabel("Lift")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

# Streamlit UI components
def show_association_analysis_page():
    st.title("Financial Metrics Association Analysis 🛒")

    # Load data
    df = load_financial_data()

    # Run association analysis
    rules = run_association_analysis(df)

    # Show results
    plot_association_graphs(rules)

    # Friendly English Insights with improved formatting
    st.write("## Insights:")
    top_rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10)
    
    # Iterate over top rules to present insights in a better way
    for _, row in top_rules.iterrows():
        antecedent = row['antecedents']
        consequent = row['consequents']
        support = row['support']
        confidence = row['confidence']
        lift = row['lift']

        # Clean up frozenset to just show the column names
        antecedent = ', '.join([str(item) for item in list(antecedent)])
        consequent = ', '.join([str(item) for item in list(consequent)])

        # Present the insights
        st.markdown(f"**🔍 Relationship Between: {antecedent} and {consequent}**")
        st.markdown(f"**💡 Insights:**")
        st.markdown(f"- This relationship occurs in **{support * 100:.1f}%** of companies.")
        st.markdown(f"- When **{antecedent}** is high, there's a **{confidence * 100:.1f}%** chance that **{consequent}** is also high.")
        st.markdown(f"- The relationship is **{lift:.2f} times** stronger than random chance.")
        st.markdown("\n---")

# Run the Streamlit app
if __name__ == "__main__":
    show_association_analysis_page()