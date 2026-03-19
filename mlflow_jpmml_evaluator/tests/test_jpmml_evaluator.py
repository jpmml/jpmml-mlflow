from mlflow_jpmml_evaluator import classpath, load_model, log_model
from mlflow_pmml.tests import _load_resource, MLFlowTest
from py4j.java_gateway import JavaGateway, JavaObject

import mlflow
import os

PMML_BYTES = _load_resource("DecisionTreeIris.pmml")

class JpmmlEvaluatorTest(MLFlowTest):

	def setUp(self):
		super().setUp()
		self._gateway = JavaGateway.launch_gateway(classpath = os.pathsep.join(classpath()))

	def tearDown(self):
		self._gateway.shutdown()
		super().tearDown()

	def test_log_load(self):
		with mlflow.start_run() as run:
			log_model(pmml = PMML_BYTES, artifact_path = "model")
		evaluator = load_model(f"runs:/{run.info.run_id}/model", jvm = self._gateway.jvm)
		self.assertIsNotNone(evaluator)
		self.assertIsInstance(evaluator, JavaObject)
