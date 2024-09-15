import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix

def main():
    df = pd.read_csv("train.csv")

    print(df.head())
    print(df.shape)

    print(df.dtypes)

    # Descriptive stats
    print(df.describe())

    # Find ans remove null vaules
    print(f"Rows containing null values: {len(df[df.isna().any(axis=1)])}")
    cols_with_null = df.columns[df.isna().any()]
    for col in cols_with_null:
        df[col] = df[col].fillna(df[col].mean())

    # Keep only valid age range (0 < age < 120)
    df = df[(df["age_in_days"] > 0) | (df["age_in_days"] < 43800)]

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Count negative income values
    if (df["Income"] < 0).sum():
        print("There are negative income values")

    # Count negatives in payments count
    negative_payments = (df["Count_3-6_months_late"] < 0).sum() + (df["Count_6-12_months_late"] < 0).sum() + (df["Count_more_than_12_months_late"] < 0).sum()

    if negative_payments > 0:
        print("There is a negative number of late payments")

    # Check for invalid percentage
    if df[(df["perc_premium_paid_by_cash_credit"] < 0) | (df["perc_premium_paid_by_cash_credit"] > 1)].size > 0:
        print("There are invalid percentages")

    # Check for invalid residence
    if len(df[~df["residence_area_type"].isin(["Urban", "Rural"])]) > 0:
        print("There are invalid residence types")

    # Converting categoric values
    df["sourcing_channel"] = df["sourcing_channel"].astype("category").cat.codes
    df["residence_area_type"] = df["residence_area_type"].astype("category").cat.codes

    # Correlation matrix
    corr_matrix = df.corr()

    # Correlation with target variable
    target_corr = corr_matrix["target"]
    print(target_corr.sort_values(ascending=False))
    plt.figure()
    sns.heatmap(corr_matrix[["target"]].sort_values(by="target", ascending=False), annot=True, cmap="coolwarm", fmt=".2f", cbar_kws={"label" : "Correlation"})
    plt.show()

    # Remove negatively correlated features
    df = df.drop(columns=["Count_6-12_months_late", "Count_3-6_months_late", "perc_premium_paid_by_cash_credit",
                         "Count_more_than_12_months_late", "sourcing_channel", "id"])
    

    x = df.drop(columns=["target"])
    y = df["target"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=31, stratify=y)

    # Decision tree
    decision_tree_model = DecisionTreeClassifier(random_state=31)
    decision_tree_model.fit(x_train, y_train)

    y_pred = decision_tree_model.predict(x_test)
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="Blues")
    plt.title("Decision Tree Confusion Matrix")
    plt.show()

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n---------- Decision Tree ----------")
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F-measure: ", f1)
    print("Accuracy: ", accuracy)

    # Naive Bayes
    nb_model = GaussianNB()
    nb_model.fit(x_train, y_train)
    y_pred_nb = nb_model.predict(x_test)

    precision = precision_score(y_test, y_pred_nb)
    recall = recall_score(y_test, y_pred_nb)
    f1 = f1_score(y_test, y_pred_nb)
    accuracy = accuracy_score(y_test, y_pred_nb)
    
    sns.heatmap(confusion_matrix(y_test, y_pred_nb), annot=True, cmap="Blues")
    plt.title("Naive Bayes Confusion Matrix")
    plt.show()
    print("\n----------- Naive Bayes -----------")
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F-measure: ", f1)
    print("Accuracy: ", accuracy)

if __name__ == "__main__":
    main()
