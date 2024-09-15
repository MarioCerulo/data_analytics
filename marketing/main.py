import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier

def read_dataset(filename: str) -> pd.DataFrame:
    """Read the dataset from a csv file"""

    return pd.read_csv(filename, delimiter=";")

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clear the data in the dataframe"""

    print(f"Rows containing null values: {len(df[df.isna().any(axis=1)])}")
    cols_with_null = df.columns[df.isna().any()]
    for col in cols_with_null:
        df[col] = df[col].fillna(df[col].mean())

    # Check for nulls
    if len(df[df.isna().any(axis=1)]) > 0:
        print("Null values found")

    # Validate the age
    invalid_age = df[(df["age"] < 18) | (df["age"] > 120)]
    if len(invalid_age) > 0:
        print(invalid_age)
        df = df[(df["age"] >= 18) & (df["age"] <= 120)]

    # Check jobs
    valid_jobs = ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired',
                  'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown']

    print(f"\nInvalid Jobs: {len(df[~df["job"].isin(valid_jobs)])}")
    df.loc[(df["job"] == "unknown"), "job"] = df["job"].mode()[0]
    
    # Check marital
    valid_marital = ["divorced", "married", "single", "unknown"]
    print(f"Invalid marital: {len(df[~df["marital"].isin(valid_marital)])}")
    df.loc[(df["marital"] == "unknown"), "marital"] = df["marital"].mode()[0] 

    # Check education
    valid_education = ["basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate", "professional.course",
                       "university.degree", "unknown"]

    print(f"Invalid Education: {len(df[~df["education"].isin(valid_education)])}")
    df.loc[(df["education"] == "unknown"), "education"] = df["education"].mode()[0]

    valid_values = ["no", "yes", "unknown"]

    # Check default
    print(f"Invalid Default: {len(df[~df["default"].isin(valid_values)])}")
    df.loc[(df["default"] == "unknown"), "default"] = df["default"].mode()[0]

    # Check housing
    print(f"Invalid Housing: {len(df[~df["housing"].isin(valid_values)])}")
    df.loc[(df["housing"] == "unknown"), "housing"] = df["housing"].mode()[0]

    # Check loan
    print(f"Invalid Loan: {len(df[~df["loan"].isin(valid_values)])}")
    df.loc[(df["loan"] == "unknown"), "loan"] = df["loan"].mode()[0]

    # Check contact
    valid_contact = ["cellular", "telephone"]
    print(f"Invalid Contact: {len(df[~df["contact"].isin(valid_contact)])}")

    # CHeck month
    valid_month = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    print(f"Invalid Month: {len(df[~df["month"].isin(valid_month)])}")

    valid_days = ["mon", "tue", "wed", "thu", "fri"]
    print(f"Invalid Days: {len(df[~df["day_of_week"].isin(valid_days)])}")

    # Check duration
    invalid_duration = df[df["duration"] < 0]
    print(f"\nInvalid duration: {len(invalid_duration)}")

    # Check campaign
    invalid_campaign = df[df["campaign"] < 0]
    print(f"Invalid campaign: {len(invalid_campaign)}")

    # Check pdays
    print(f"Invalid pdays: {len(df[df["pdays"] < 0])}")

    # Check previous
    print(f"Invalid previous: {len(df[df["previous"] < 0])}")

    # Check poutcome
    valid_poutcome = ["nonexistent", "success", "failure"]
    print(f"Invalid poutcome: {len(df[~df["poutcome"].isin(valid_poutcome)])}")

    return df

def correlation(df: pd.DataFrame):
    """Plot the correlation matrix of the target variable"""

    # Convert categories to ints
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype("category").cat.codes
    
    # Plot the correlation matrix
    corr_matrix = df.corr()
    plt.figure()
    sns.heatmap(corr_matrix[["y"]].sort_values(by="y", ascending=False), annot=True)
    plt.show()

def decision_tree(X_train, X_test, y_train, y_test):
    """Train and test a decision tree"""

    dt = DecisionTreeClassifier(random_state=31)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
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

def naive_bayes(X_train, X_test, y_train, y_test):
    """Train and test a naive bayesian model"""

    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)

    precision = precision_score(y_test, y_pred_nb)
    recall = recall_score(y_test, y_pred_nb)
    f1 = f1_score(y_test, y_pred_nb)
    accuracy = accuracy_score(y_test, y_pred_nb)
    
    plt.figure()
    sns.heatmap(confusion_matrix(y_test, y_pred_nb), annot=True, cmap="Blues")
    plt.title("Naive Bayes Confusion Matrix")
    plt.show()
    print("\n----------- Naive Bayes -----------")
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F-measure: ", f1)
    print("Accuracy: ", accuracy)

def stats(df):
    print(df.describe())

def main():
    df = read_dataset("dataset.csv")
    print(f"{df.shape}\n\n{df.head()}\n\n{df.dtypes}\n")

    stats(df)

    df = clean_data(df)

    correlation(df)

    # Drop the columns negatively related to the target
    df.drop(columns=["nr.employed", "pdays", "euribor3m", "emp.var.rate", "contact", "cons.price.idx", "default",
                     "campaign", "month", "loan"], inplace=True)

    # Separate the target column from the dataset
    X = df.drop(columns=["y"])
    y = df["y"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=31, stratify=y)
    print()
    decision_tree(X_train, X_test, y_train, y_test)
    naive_bayes(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
