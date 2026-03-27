from jpmml_mlflow.tests import _load_iris
from sklearn.tree import DecisionTreeClassifier

def _make_sk_model():
	iris_X, iris_y = _load_iris()

	model = DecisionTreeClassifier(max_depth = 3, random_state = 42)
	model.fit(iris_X, iris_y)

	return model
