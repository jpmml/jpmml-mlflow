from jpmml_mlflow.tests import _find_resource
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

import pandas

def _make_sk_model():
	df = pandas.read_csv(_find_resource("Iris.csv"))

	iris_X = df[df.columns[0:4]]
	iris_y = df["Species"]

	model = DecisionTreeClassifier(max_depth = 3, random_state = 42)
	model.fit(iris_X, iris_y)

	return model
