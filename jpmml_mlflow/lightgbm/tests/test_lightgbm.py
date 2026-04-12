from jpmml_mlflow.tests import MLflowTest
from jpmml_mlflow.lightgbm import log_model
from jpmml_mlflow.lightgbm.tests import _make_lgb_booster, _make_lgb_model

import mlflow

class LightGBMTest(MLflowTest):

	def test_lgb_booster(self):
		lgb_booster = _make_lgb_booster()
		with mlflow.start_run() as run:
			log_model(lgb_booster, artifact_path = "model")
		self.assertFlavors(run, ["lightgbm", "pmml"])

	def test_lgb_model(self):
		lgb_model, signature, input_example = _make_lgb_model(with_names = False)
		with mlflow.start_run() as run:
			log_model(lgb_model, artifact_path = "model", signature = signature, input_example = input_example)
		self.assertFlavors(run, ["lightgbm", "pmml"])
		self.assertIrisSignature(run)
		self.assertIrisInputExample(run, labels = lgb_model.classes_)
