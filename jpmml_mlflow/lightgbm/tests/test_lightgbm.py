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
		lgb_model = _make_lgb_model()
		with mlflow.start_run() as run:
			log_model(lgb_model, artifact_path = "model")
		self.assertFlavors(run, ["lightgbm", "pmml"])
