import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from csv import reader
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

def main():
    groceries = []
    
    # Reading the dataset
    with open("dataset.csv", "r") as read_obj:
        csv_reader = reader(read_obj)
    
        for row in csv_reader:
            groceries.append([item for item in row if item])

    # Fitting the list and converting the transaction to true and false
    encoder = TransactionEncoder()
    transactions = encoder.fit(groceries).transform(groceries)
    
    # Converting true and false into 1 and 0
    transactions = transactions.astype("int")
    
    # Creating the dataframe
    df = pd.DataFrame(transactions, columns=encoder.columns_)
    df.drop_duplicates(inplace=True)

    print(df.head())
    print(df.shape)

    # Apriori algorithm
    frequent_itemsets = apriori(df, min_support=0.01, use_colnames=True)

    frequent_itemsets["length"] = frequent_itemsets["itemsets"].apply(lambda x: len(x))

    frequent_itemsets = frequent_itemsets.sort_values(by="support", ascending=False)
    print(frequent_itemsets[(frequent_itemsets["length"] == 1) & (frequent_itemsets["support"] >= 0.02)] [0:5])

    rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)

    print(rules.sort_values(by="confidence", ascending=False)[0:10])
    print(rules[(rules["support"] >= 0.01) & (rules["lift"] > 1.0)])

if __name__ == "__main__":
    main()
