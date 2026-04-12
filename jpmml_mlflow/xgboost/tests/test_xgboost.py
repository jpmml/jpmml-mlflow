from jpmml_mlflow.tests import MLflowTest
from jpmml_mlflow.xgboost import log_model
from jpmml_mlflow.xgboost.tests import _make_xgb_booster, _make_xgb_model

import mlflow

class XGBoostTest(MLflowTest):

	def test_xgb_booster(self):
		xgb_booster, fmap = _make_xgb_booster()
		with mlflow.start_run() as run:
			log_model(xgb_booster, artifact_path = "model", fmap = fmap)
		self.assertFlavors(run, ["xgboost", "pmml"])
		self.assertSignature(run, required_fields = ["Sepal_Length", "Petal_Length"], prohibited_fields = ["x1", "x2", "x3", "x4"])

	def test_xgb_model(self):
		xgb_model, signature, input_example = _make_xgb_model(with_names = False)
		with mlflow.start_run() as run:
			log_model(xgb_model, signature = signature, input_example = input_example, artifact_path = "model")
		self.assertFlavors(run, ["xgboost", "pmml"])
		self.assertIrisSignature(run)
		self.assertIrisInputExample(run, labels = xgb_model.classes_)
