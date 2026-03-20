from jpmml_mlflow.tests import MLFlowTest
from jpmml_mlflow.sklearn import log_model
from jpmml_mlflow.sklearn.tests import _make_sk_model
from mlflow.models import Model

import mlflow

class SkLearnTest(MLFlowTest):

	def test_decision_tree_iris(self):
		sk_model = _make_sk_model()
		with mlflow.start_run() as run:
			log_model(sk_model, artifact_path = "model")
		mlflow_sk_model = Model.load(f"runs:/{run.info.run_id}/model")
		self.assertIn("sklearn", mlflow_sk_model.flavors)
		self.assertIn("pmml", mlflow_sk_model.flavors)
