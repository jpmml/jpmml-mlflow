from jpmml_mlflow.tests import _load_iris
from mlflow.models import infer_signature
from sklearn.tree import DecisionTreeClassifier

def _make_sk_model(with_signature = False):
	iris_X, iris_y = _load_iris()

	model = DecisionTreeClassifier(max_depth = 3, random_state = 42)

	if with_signature:
		signature = infer_signature(iris_X, iris_y)
		input_example = iris_X.sample(n = 10, random_state = 42)

		model.fit(iris_X, iris_y)
		return (model, signature, input_example)
	else:
		model.fit(iris_X.values, iris_y.values)
		return model
