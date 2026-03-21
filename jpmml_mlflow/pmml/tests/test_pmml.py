from jpmml_mlflow.tests import _find_resource, _load_resource, MLflowTest
from jpmml_mlflow.pmml import load_model, log_model
from pathlib import Path

import mlflow

PMML_BYTES = _load_resource("DecisionTreeIris.pmml")

class PMMLTest(MLflowTest):

	def test_bytes(self):
		with mlflow.start_run() as run:
			log_model(pmml = PMML_BYTES, artifact_path = "model")
		pmml_bytes = load_model(f"runs:/{run.info.run_id}/model")
		self.assertEqual(PMML_BYTES, pmml_bytes)

	def test_path(self):
		pmml_path = Path(_find_resource("DecisionTreeIris.pmml"))
		with mlflow.start_run() as run:
			log_model(pmml = pmml_path, artifact_path = "model")
		pmml_bytes = load_model(f"runs:/{run.info.run_id}/model")
		self.assertEqual(PMML_BYTES, pmml_bytes)
