import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Function to run Apriori and generate association rules
def run_apriori(dataset, min_support, min_confidence):
    # Apply Apriori algorithm
    frequent_itemsets = apriori(dataset, min_support=min_support, use_colnames=True)
    
    # Generate association rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    
    # Evaluate rules based on support, confidence, and lift
    return frequent_itemsets, rules

# =====================================
# Dataset 1: Groceries Dataset
# =====================================
print("\n--- Dataset 1: Groceries ---")
url_groceries = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/groceries.csv"
groceries_df = pd.read_csv(url_groceries, header=None)
groceries_df.columns = ['Items']
groceries_df['TransactionID'] = groceries_df.index

# Convert the dataset into the required format
groceries_transactions = groceries_df.groupby('TransactionID')['Items'].apply(lambda x: list(x)).tolist()
te = TransactionEncoder()
te_groceries = te.fit_transform(groceries_transactions)
groceries_encoded = pd.DataFrame(te_groceries, columns=te.columns_)

# Running Apriori with different parameters
print("\nResults with min_support=0.5 and min_confidence=0.75")
frequent_items_g1, rules_g1 = run_apriori(groceries_encoded, min_support=0.5, min_confidence=0.75)
print("\nFrequent Itemsets:\n", frequent_items_g1.head())
print("\nAssociation Rules:\n", rules_g1[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())

print("\nResults with min_support=0.6 and min_confidence=0.6")
frequent_items_g2, rules_g2 = run_apriori(groceries_encoded, min_support=0.6, min_confidence=0.6)
print("\nFrequent Itemsets:\n", frequent_items_g2.head())
print("\nAssociation Rules:\n", rules_g2[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())

# =====================================
# Dataset 2: Online Retail Dataset
# =====================================
print("\n--- Dataset 2: Online Retail ---")
url_retail = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
retail_df = pd.read_excel(url_retail)
retail_df.dropna(subset=['InvoiceNo', 'Description'], inplace=True)
retail_df = retail_df[~retail_df['InvoiceNo'].astype(str).str.startswith('C')]  # Remove canceled transactions

# Convert to transaction format
basket = retail_df.groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack().reset_index().fillna(0)
basket.set_index('InvoiceNo', inplace=True)
basket = basket.applymap(lambda x: 1 if x > 0 else 0)

# Running Apriori with different parameters
print("\nResults with min_support=0.5 and min_confidence=0.75")
frequent_items_r1, rules_r1 = run_apriori(basket, min_support=0.5, min_confidence=0.75)
print("\nFrequent Itemsets:\n", frequent_items_r1.head())
print("\nAssociation Rules:\n", rules_r1[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())

print("\nResults with min_support=0.6 and min_confidence=0.6")
frequent_items_r2, rules_r2 = run_apriori(basket, min_support=0.6, min_confidence=0.6)
print("\nFrequent Itemsets:\n", frequent_items_r2.head())
print("\nAssociation Rules:\n", rules_r2[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())
