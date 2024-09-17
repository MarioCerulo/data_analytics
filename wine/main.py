import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt
from imblearn.over_sampling import SMOTE

def load_dataset() -> pd.DataFrame:
	"""Load the wine dataset and create a pandas DataFrame"""

	wine = load_wine()
	return pd.DataFrame(data=np.c_[wine["data"], wine["target"]], columns=wine["feature_names"] + ["target"])

def correlation_matrix(df: pd.DataFrame) -> None:
	"""Plot the correlation matrix"""

	corr_mat = df.corr()

	plt.figure(figsize=[15, 15])
	plt.title("Correlation Matrix")
	sns.heatmap(corr_mat, annot=True, cmap="GnBu", robust=True, fmt=".2f")
	plt.show()
	
def remove_unrelated_variability_with_target(df: pd.DataFrame, target: str, threshold: float) -> pd.DataFrame:
	corr = df.corr()
	corr_target = abs(corr[target])

	relevant_features = corr_target[corr_target > threshold]

	print(f"Relevant features:\n{relevant_features}\n")

	relevant_features_col = relevant_features.keys().tolist()

	return df[relevant_features_col]

def classification(df: pd.DataFrame, target: str) -> None:
	"""Train and test a decision tree on the dataframe"""

	X = df.drop(columns=[target])
	y = df[target]

	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=31, train_size=0.2, shuffle=True)
	print(f"y_train counts before:\n{y_train.value_counts()}\n")
	print(f"y_test counts before:\n{y_test.value_counts()}\n")

	smote = SMOTE(sampling_strategy="not majority", random_state=31)
	X_train, y_train = smote.fit_resample(X_train, y_train)
	X_test, y_test = smote.fit_resample(X_test, y_test)
	print(f"y_train counts after:\n{y_train.value_counts()}\n")
	print(f"y_test counts after:\n{y_test.value_counts()}\n")

	classifier = DecisionTreeClassifier()
	classifier.fit(X_train, y_train)
	y_pred = classifier.predict(X_test)

	plt.hist([y_pred, y_test], bins=[0, 1, 2, 3, 4, 5], label=["y_pred", "y_test"], align="left")
	plt.xticks([0, 1, 2])
	plt.yscale("linear")
	plt.title("Decision Tree Classifier")
	plt.legend()
	plt.show()

def main():
	df = load_dataset()

	print(f"{df.describe()}\n")

	print(f"{df.dtypes}\n")
	print(f"Null values:\n{df.isna().sum()}\n")

    # Target values
	print(f"Target values: {df["target"].unique()}\n")
	print(f"Value count:\n{df["target"].value_counts()}\n")
    
	correlation_matrix(df)

	target = "target"
	df = remove_unrelated_variability_with_target(df, target, 0.45)
	classification(df, target)

if __name__ == "__main__":
	main()
