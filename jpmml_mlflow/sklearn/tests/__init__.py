from jpmml_mlflow.tests import _load_iris
from mlflow.models import infer_signature
from sklearn.tree import DecisionTreeClassifier

def _make_sk_model(with_names = False):
	iris_X, iris_y = _load_iris()
	signature = infer_signature(iris_X, iris_y)
	input_example = iris_X.sample(n = 10, random_state = 42)

	model = DecisionTreeClassifier(max_depth = 3, random_state = 42)

	if with_names:
		model.fit(iris_X, iris_y)
	else:
		model.fit(iris_X.values, iris_y.values)

	return (model, signature, input_example)