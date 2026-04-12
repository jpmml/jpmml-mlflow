from jpmml_mlflow.sklearn.tests import _load_iris
from lightgbm import Dataset, LGBMClassifier
from mlflow.models import infer_signature

import lightgbm

def _make_lgb_booster():
	iris_X, iris_y = _load_iris()
	# Encode labels from string to numeric
	iris_y = iris_y.astype("category").cat.codes
	iris_ds = Dataset(iris_X, label = iris_y)

	lgb_params = {
		"max_depth" : 3,
		"num_iterations" : 7
	}
	booster = lightgbm.train(lgb_params, iris_ds)

	return booster

def _make_lgb_model(with_names = False):
	iris_X, iris_y = _load_iris()
	signature = infer_signature(iris_X, iris_y)
	input_example = iris_X.sample(n = 10, random_state = 42)

	model = LGBMClassifier(max_depth = 3, n_estimators = 7, random_state = 42)

	if with_names:
		model.fit(iris_X, iris_y)
	else:
		model.fit(iris_X.values, iris_y.values)

	return (model, signature, input_example)