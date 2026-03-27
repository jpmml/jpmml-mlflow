from jpmml_mlflow.tests import MLflowTest
from jpmml_mlflow.sklearn import log_model
from jpmml_mlflow.sklearn.tests import _make_sk_model
from mlflow.artifacts import download_artifacts
from mlflow.models import Model

import mlflow

class SkLearnTest(MLflowTest):

	def test_sk_model(self):
		sk_model = _make_sk_model()
		with mlflow.start_run() as run:
			log_model(sk_model, artifact_path = "model")

		model_path = download_artifacts(f"runs:/{run.info.run_id}/model")
		mlflow_sk_model = Model.load(model_path)
		self.assertIn("sklearn", mlflow_sk_model.flavors)
		self.assertIn("pmml", mlflow_sk_model.flavors)
