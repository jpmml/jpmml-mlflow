from jpmml_mlflow.tests import PySparkTest
from jpmml_mlflow.spark import classpath, log_model
from jpmml_mlflow.spark.tests import _make_spark_model
from mlflow.models import Model
from typing import List

import mlflow

class SparkTest(PySparkTest):

	@classmethod
	def _spark_jars(cls) -> List[str]:
		return classpath()

	def test_decision_tree_iris(self):
		spark_model, df = _make_spark_model(self._spark)
		with mlflow.start_run() as run:
			log_model(spark_model, artifact_path = "model", input_example = df.sample(fraction = 0.1))
		mlflow_spark_model = Model.load(f"runs:/{run.info.run_id}/model")
		self.assertIn("spark", mlflow_spark_model.flavors)
		self.assertIn("pmml", mlflow_spark_model.flavors)
