from jpmml_mlflow.tests import MLflowTest
from jpmml_mlflow.xgboost import log_model
from jpmml_mlflow.xgboost.tests import _make_xgb_booster, _make_xgb_model

import jpmml_mlflow.pmml
import mlflow

class XGBoostTest(MLflowTest):

	def test_xgb_booster(self):
		xgb_booster, fmap = _make_xgb_booster()
		with mlflow.start_run() as run:
			log_model(xgb_booster, artifact_path = "model", fmap = fmap)
		self.assertFlavors(run, ["xgboost", "pmml"])

		pmml_bytes = jpmml_mlflow.pmml.load_model(f"runs:/{run.info.run_id}/model")
		pmml_str = pmml_bytes.decode("utf-8")
		self.assertIn("Sepal_Length", pmml_str)
		self.assertIn("Petal_Length", pmml_str)
		self.assertNotIn("x1", pmml_str)

	def test_xgb_model(self):
		xgb_model = _make_xgb_model()
		with mlflow.start_run() as run:
			log_model(xgb_model, artifact_path = "model")
		self.assertFlavors(run, ["xgboost", "pmml"])
