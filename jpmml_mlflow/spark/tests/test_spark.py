from jpmml_mlflow.tests import PySparkTest
from jpmml_mlflow.spark import classpath, log_model
from jpmml_mlflow.spark.tests import _make_spark_model
from typing import List

import mlflow

class SparkTest(PySparkTest):

	@classmethod
	def _spark_jars(cls) -> List[str]:
		return classpath()

	def test_classpath(self):
		spark34_jars = classpath("3.4.")
		spark35_jars = classpath("3.5.")
		spark40_jars = classpath("4.0.")
		spark41_jars = classpath("4.1.")

		self.assertEqual(3 + 15, len(spark34_jars))
		self.assertEqual(3 + 15, len(spark35_jars))
		self.assertEqual(3 + 15, len(spark40_jars))
		self.assertEqual(3 + 15, len(spark41_jars))
		self.assertNotEqual(set(spark34_jars[0:3]), set(spark41_jars[0:3]))
		self.assertEqual(set(spark34_jars[3:]), set(spark41_jars[3:]))

	def test_spark_model(self):
		spark_model, _, input_example = _make_spark_model(self._spark)
		with mlflow.start_run() as run:
			log_model(spark_model, input_example = input_example, artifact_path = "model")
		self.assertFlavors(run, ["spark", "pmml"])
		self.assertIrisSignature(run)
		self.assertIrisInputExample(run)
