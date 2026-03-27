from jpmml_mlflow.tests import MLflowTest
from jpmml_mlflow.lightgbm import log_model
from jpmml_mlflow.lightgbm.tests import _make_lgb_booster, _make_lgb_model
from mlflow.artifacts import download_artifacts
from mlflow.models import Model

import mlflow

class LightGBMTest(MLflowTest):

	def _workflow(self, lgb_model):
		with mlflow.start_run() as run:
			log_model(lgb_model, artifact_path = "model")

		model_path = download_artifacts(f"runs:/{run.info.run_id}/model")
		mlflow_lgb_model = Model.load(model_path)
		self.assertIn("lightgbm", mlflow_lgb_model.flavors)
		self.assertIn("pmml", mlflow_lgb_model.flavors)

	def test_lgb_booster(self):
		lgb_booster = _make_lgb_booster()
		self._workflow(lgb_booster)

	def test_lgb_model(self):
		lgb_model = _make_lgb_model()
		self._workflow(lgb_model)
