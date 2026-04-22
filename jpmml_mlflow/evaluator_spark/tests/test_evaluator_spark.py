from jpmml_evaluator_pyspark import FlatPMMLTransformer, NestedPMMLTransformer
from jpmml_mlflow.tests import _find_resource, _load_resource, PySparkTest
from jpmml_mlflow.evaluator_spark import spark_jars, load_model, log_model

import mlflow

PMML_BYTES = _load_resource("DecisionTreeIris.pmml")

class JPMMLEvaluatorSparkTest(PySparkTest):

	@classmethod
	def _spark_jars(cls) -> str:
		return spark_jars()

	def test_log_load(self):
		jvm = self._spark._jvm
		with mlflow.start_run() as run:
			log_model(pmml = PMML_BYTES, artifact_path = "model")

		df = self._spark.read.csv(_find_resource("Iris.csv"), header = True, inferSchema = True)
		df = df.drop("Species")

		pmmlTransformer = load_model(f"runs:/{run.info.run_id}/model", jvm = jvm)
		self.assertIsInstance(pmmlTransformer, FlatPMMLTransformer)

		pmml_df = pmmlTransformer.transform(df)
		self.assertEqual(150, pmml_df.count())
		self.assertEqual(4 + (1 + 3) + 1, len(pmml_df.columns))

		pmmlTransformer = load_model(f"runs:/{run.info.run_id}/model", transformer_type = NestedPMMLTransformer, jvm = jvm)
		self.assertIsInstance(pmmlTransformer, NestedPMMLTransformer)

		pmml_df = pmmlTransformer.transform(df)
		self.assertEqual(150, pmml_df.count())
		self.assertEqual(4 + (1) + 1, len(pmml_df.columns))
