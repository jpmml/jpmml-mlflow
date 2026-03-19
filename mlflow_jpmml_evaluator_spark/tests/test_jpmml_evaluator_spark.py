from jpmml_mlflow.tests import MLFlowTest
from mlflow_jpmml_evaluator_spark import classpath, load_model, log_model
from mlflow_pmml.tests import _load_resource
from py4j.java_gateway import JavaObject
from pyspark.sql import SparkSession

import mlflow

PMML_BYTES = _load_resource("DecisionTreeIris.pmml")

class JPMMLEvaluatorSparkTest(MLFlowTest):

	def setUp(self):
		self._spark = SparkSession.builder \
			.master("local") \
			.config("spark.jars", ",".join(classpath())) \
			.getOrCreate()
		super().setUp()

	def tearDown(self):
		super().tearDown()
		self._spark.stop()

	def test_classpath(self):
		spark3_jars = classpath("3.X")
		spark4_jars = classpath("4.X")
		self.assertEqual(1 + 15, len(spark3_jars))
		self.assertEqual(1 + 15, len(spark4_jars))
		self.assertNotEqual(spark3_jars[0], spark4_jars[0])
		self.assertEqual(set(spark3_jars[1:]), set(spark4_jars[1:]))

	def test_log_load(self):
		jvm = self._spark._jvm
		with mlflow.start_run() as run:
			log_model(pmml = PMML_BYTES, artifact_path = "model")
		evaluator = load_model(f"runs:/{run.info.run_id}/model", jvm = jvm)
		self.assertIsNotNone(evaluator)
		self.assertIsInstance(evaluator, JavaObject)

		flatPmmlTransformer = jvm.org.jpmml.evaluator.spark.FlatPMMLTransformer(evaluator)
		self.assertEqual(1 + 3, (flatPmmlTransformer.pmmlFields()).size())

		nestedPmmlTransformer = jvm.org.jpmml.evaluator.spark.NestedPMMLTransformer(evaluator)
		self.assertEqual(1 + 3, (nestedPmmlTransformer.pmmlFields()).size())
