from jpmml_mlflow.tests import MLflowTest
from jpmml_mlflow.sklearn import log_model
from jpmml_mlflow.sklearn.tests import _make_sk_model

import mlflow

class SkLearnTest(MLflowTest):

	def test_sk_model(self):
		sk_model = _make_sk_model(with_signature = False)
		with mlflow.start_run() as run:
			log_model(sk_model, artifact_path = "model")
		self.assertFlavors(run, ["sklearn", "pmml"])
		self.assertSignature(run, required_fields = ["y", "x3", "x4"])

	def test_sk_model_with_signature(self):
		sk_model, signature, input_example = _make_sk_model(with_signature = True)
		with mlflow.start_run() as run:
			log_model(sk_model, signature = signature, input_example = input_example, artifact_path = "model")
		self.assertFlavors(run, ["sklearn", "pmml"])
		self.assertSignature(run, required_fields = ["Species", "Petal_Length", "Petal_Width"])
		self.assertInputExample(run, ["Petal_Length", "Petal_Width"] + ["probability({})".format(label) for label in ["setosa", "versicolor", "virginica"]])
