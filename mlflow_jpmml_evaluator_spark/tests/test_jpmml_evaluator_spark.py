from mlflow_jpmml_evaluator_spark import classpath, load_model, log_model
from mlflow_pmml.tests import MLFlowTest, _load_resource
from py4j.java_gateway import JavaObject
from pyspark.sql import SparkSession

import mlflow

PMML_BYTES = _load_resource("DecisionTreeIris.pmml")

class JpmmlEvaluatorSparkTest(MLFlowTest):

	def setUp(self):
		self._spark = SparkSession.builder \
			.master("local") \
			.config("spark.jars", ",".join(classpath())) \
			.getOrCreate()
		super().setUp()

	def tearDown(self):
		super().tearDown()
		self._spark.stop()

	def test_log_load(self):
		jvm = self._spark._jvm
		with mlflow.start_run() as run:
			log_model(pmml = PMML_BYTES, artifact_path = "model")
		evaluator = load_model(f"runs:/{run.info.run_id}/model", jvm = jvm)
		self.assertIsNotNone(evaluator)
		self.assertIsInstance(evaluator, JavaObject)
