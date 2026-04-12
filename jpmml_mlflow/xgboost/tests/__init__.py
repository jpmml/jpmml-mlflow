from jpmml_mlflow.sklearn.tests import _load_iris
from mlflow.models import infer_signature
from sklearn2pmml.xgboost import make_feature_map
from xgboost import DMatrix, XGBClassifier

import xgboost

def _make_xgb_booster():
	iris_X, iris_y = _load_iris()
	iris_fmap = make_feature_map(iris_X)
	# Anonymize columns
	iris_X = iris_X.values
	# Encode labels from string to numeric
	iris_y = iris_y.astype("category").cat.codes
	iris_dm = DMatrix(iris_X, label = iris_y)

	xgb_params = {
		"max_depth" : 3
	}
	booster = xgboost.train(xgb_params, iris_dm, num_boost_round = 7)

	return (booster, iris_fmap)

def _make_xgb_model(with_names = False):
	iris_X, iris_y = _load_iris()
	signature = infer_signature(iris_X, iris_y)
	input_example = iris_X.sample(n = 10, random_state = 42)
	# Encode labels from string to numeric
	iris_y = iris_y.astype("category").cat.codes

	model = XGBClassifier(max_depth = 3, n_estimators = 7, random_state = 42)

	if with_names:
		model.fit(iris_X, iris_y)
	else:
		model.fit(iris_X.values, iris_y.values)

	return (model, signature, input_example)