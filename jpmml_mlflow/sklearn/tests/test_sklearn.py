from jpmml_mlflow.tests import MLflowTest
from jpmml_mlflow.sklearn import log_model
from jpmml_mlflow.sklearn.tests import _make_sk_model

import mlflow

class SkLearnTest(MLflowTest):

	def test_sk_model(self):
		sk_model = _make_sk_model()
		with mlflow.start_run() as run:
			log_model(sk_model, artifact_path = "model")
		self.assertFlavors(run, ["sklearn", "pmml"])
