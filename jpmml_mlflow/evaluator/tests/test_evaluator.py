from jpmml_mlflow.tests import MLFlowTest
from jpmml_mlflow.evaluator import classpath, load_model, log_model
from jpmml_mlflow.pmml.tests import _load_resource
from py4j.java_gateway import JavaGateway, JavaObject

import mlflow
import os

PMML_BYTES = _load_resource("DecisionTreeIris.pmml")

class JPMMLEvaluatorTest(MLFlowTest):

	def setUp(self):
		super().setUp()
		self._gateway = JavaGateway.launch_gateway(classpath = os.pathsep.join(classpath()))

	def tearDown(self):
		self._gateway.shutdown()
		super().tearDown()

	def test_classpath(self):
		jars = classpath()
		self.assertEqual(15, len(jars))

	def test_log_load(self):
		with mlflow.start_run() as run:
			log_model(pmml = PMML_BYTES, artifact_path = "model")
		evaluator = load_model(f"runs:/{run.info.run_id}/model", jvm = self._gateway.jvm)
		self.assertIsNotNone(evaluator)
		self.assertIsInstance(evaluator, JavaObject)
