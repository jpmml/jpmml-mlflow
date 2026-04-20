from jpmml_mlflow.tests import PySparkTest
from jpmml_mlflow.spark import spark_jars, log_model
from jpmml_mlflow.spark.tests import _make_spark_model

import mlflow

class SparkTest(PySparkTest):

	@classmethod
	def _spark_jars(cls) -> str:
		return spark_jars()

	def test_spark_model(self):
		spark_model, _, input_example = _make_spark_model(self._spark)
		with mlflow.start_run() as run:
			log_model(spark_model, artifact_path = "model", input_example = input_example)
		self.assertFlavors(run, ["spark", "pmml"])
		self.assertIrisSignature(run)
		self.assertIrisInputExample(run)
