from mlflow_pmml import load_model, log_model
from mlflow_pmml.tests import _load_resource, MLFlowTest

import mlflow

PMML_BYTES = _load_resource("DecisionTreeIris.pmml")

class PMMLTest(MLFlowTest):

	def test_bytes(self):
		with mlflow.start_run() as run:
			log_model(pmml = PMML_BYTES, artifact_path = "model")
		pmml_bytes = load_model(f"runs:/{run.info.run_id}/model")
		self.assertEqual(PMML_BYTES, pmml_bytes)

	def test_str(self):
		pmml_str = PMML_BYTES.decode("utf-8")
		with mlflow.start_run() as run:
			log_model(pmml = pmml_str, artifact_path = "model")
		pmml_bytes = load_model(f"runs:/{run.info.run_id}/model")
		self.assertEqual(PMML_BYTES, pmml_bytes)
