from mlflow_pmml import load_model, log_model
from unittest import TestCase

import mlflow
import shutil
import tempfile

PMML_BYTES = b"<PMML/>"

class MLFlowTest(TestCase):

	def setUp(self):
		self._tracking_dir = tempfile.mkdtemp()
		mlflow.set_tracking_uri(f"file://{self._tracking_dir}")
		mlflow.set_experiment("test")

	def tearDown(self):
		mlflow.set_tracking_uri(None)
		shutil.rmtree(self._tracking_dir)

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
