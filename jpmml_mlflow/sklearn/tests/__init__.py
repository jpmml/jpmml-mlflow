from jpmml_mlflow.tests import _load_iris
from mlflow.models import infer_signature
from sklearn.tree import DecisionTreeClassifier

def _make_sk_model(with_signature = False):
	iris_X, iris_y = _load_iris()

	model = DecisionTreeClassifier(max_depth = 3, random_state = 42)

	if with_signature:
		signature = infer_signature(iris_X, iris_y)

		model.fit(iris_X, iris_y)
		return (model, signature)
	else:
		model.fit(iris_X.values, iris_y.values)
		return model
