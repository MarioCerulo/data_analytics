import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

def read_dataset(filename: str):
    df = pd.read_csv(filename)

    print(f"\n{df.head()}")
    print(f"\n{df.dtypes}")

    return df

def data_cleaning(df: pd.DataFrame):
    # Check for null data
    if len(df[df.isna().any(axis=1)]) > 0:
        print("Null data found")
        df.dropna(inplace=True)

    df.drop_duplicates()

    # Check for invalid negatives
    negative_steps = len(df[df["step"] < 0])
    if negative_steps > 0:
        print(f"Negative step values found: {negative_steps}")

    print(f"\nTransaction types: {df["type"].unique()}")

    negative_amounts = len(df[df["amount"] < 0])
    if negative_amounts > 0:
        print(f"\nNegative amounts found: {negative_amounts}")

    negative_old_balance_orig = len(df[df["oldbalanceOrg"] < 0])
    if negative_old_balance_orig > 0:
        print(f"\nNegative old balance found: {negative_old_balance_orig}")

    negative_new_balance_orig = len(df[df["newbalanceOrig"] < 0])
    if negative_new_balance_orig > 0:
        print(f"\nNegative new balance found: {negative_new_balance_orig}")

    negative_old_balance_dest = len(df[df["oldbalanceDest"] < 0])
    if negative_old_balance_dest > 0:
        print(f"\nNegative old balance found: {negative_old_balance_dest}")

    negative_old_balance_dest = len(df[df["oldbalanceDest"] < 0])
    if negative_old_balance_dest > 0:
        print(f"\nNegative old balance found: {negative_old_balance_dest}")

    return df

def target_corr(df: pd.DataFrame, target: str):

    corr_matrix = df.corr()
    target_corr_matrix = corr_matrix[[target]].sort_values(by=target, ascending=False)

    plt.figure()
    sns.heatmap(target_corr_matrix, annot=True)
    plt.title("Correlation")
    plt.show()


def decision_tree(X_train: pd.Series, X_test: pd.Series, y_train: pd.Series, y_test: pd.Series):
    decision_tree_model = DecisionTreeClassifier(random_state=31)
    decision_tree_model.fit(X_train, y_train)
    y_pred = decision_tree_model.predict(X_test)

    plt.figure()
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


def naive_bayes(X_train: pd.Series, X_test: pd.Series, y_train: pd.Series, y_test: pd.Series):
    nb = GaussianNB()

    nb.fit(X_train, y_train)

    y_pred_nb = nb.predict(X_test)

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

def main():
    # Load the dataset
    df = read_dataset("dataset.csv")

    # Clean the dataset
    df = data_cleaning(df)
    
    # Convert categoric values into float
    df["type"] = df["type"].astype("category").cat.codes
    df["nameOrig"] = df["nameOrig"].astype("category").cat.codes
    df["nameDest"] = df["nameDest"].astype("category").cat.codes

    target = "isFraud"

    target_corr(df, target)

    df.drop(columns=["nameOrig", "nameDest", "oldbalanceDest", "newbalanceOrig", "newbalanceDest"], inplace=True)

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=31, stratify=y)

    decision_tree(X_train, X_test, y_train, y_test)

    naive_bayes(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
