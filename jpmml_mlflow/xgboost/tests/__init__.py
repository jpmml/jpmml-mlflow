from jpmml_mlflow.sklearn.tests import _load_iris
from xgboost import DMatrix, XGBClassifier

import xgboost

def _make_xgb_booster():
	iris_X, iris_y = _load_iris()
	# Encode labels from string to numeric
	iris_y = iris_y.astype("category").cat.codes
	iris_dm = DMatrix(iris_X, label = iris_y)

	xgb_params = {
		"max_depth" : 3,
		"num_boost_round" : 7
	}
	booster = xgboost.train(xgb_params, iris_dm)

	return booster

def _make_xgb_model():
	iris_X, iris_y = _load_iris()
	# Encode labels from string to numeric
	iris_y = iris_y.astype("category").cat.codes

	model = XGBClassifier(max_depth = 3, n_estimators = 7, random_state = 42)
	model.fit(iris_X, iris_y)

	return model
