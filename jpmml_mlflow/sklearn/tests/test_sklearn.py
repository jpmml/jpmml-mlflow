from jpmml_mlflow.tests import MLflowTest
from jpmml_mlflow.sklearn import log_model
from jpmml_mlflow.sklearn.tests import _make_sk_model

import mlflow

class SkLearnTest(MLflowTest):

	def test_sk_model_default(self):
		sk_model, _, _ = _make_sk_model(with_names = False)
		with mlflow.start_run() as run:
			log_model(sk_model, artifact_path = "model")
		self.assertFlavors(run, ["sklearn", "pmml"])
		self.assertSignature(run, required_fields = ["y", "x3", "x4"])

	def test_sk_model(self):
		sk_model, signature, input_example = _make_sk_model(with_names = False)
		with mlflow.start_run() as run:
			log_model(sk_model, artifact_path = "model", signature = signature, input_example = input_example)
		self.assertFlavors(run, ["sklearn", "pmml"])
		self.assertIrisSignature(run)
		self.assertIrisInputExample(run, labels = sk_model.classes_)
