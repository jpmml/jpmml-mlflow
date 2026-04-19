from jpmml_evaluator_pyspark import FlatPMMLTransformer, NestedPMMLTransformer
from jpmml_mlflow.tests import _find_resource, _load_resource, PySparkTest
from jpmml_mlflow.evaluator_spark import classpath, load_model, log_model
from py4j.java_gateway import JavaObject
from typing import List

import mlflow

PMML_BYTES = _load_resource("DecisionTreeIris.pmml")

class JPMMLEvaluatorSparkTest(PySparkTest):

	@classmethod
	def _spark_jars(cls) -> List[str]:
		return classpath()

	def test_log_load(self):
		jvm = self._spark._jvm
		with mlflow.start_run() as run:
			log_model(pmml = PMML_BYTES, artifact_path = "model")
		evaluator = load_model(f"runs:/{run.info.run_id}/model", jvm = jvm)
		self.assertIsNotNone(evaluator)
		self.assertIsInstance(evaluator, JavaObject)

		df = self._spark.read.csv(_find_resource("Iris.csv"), header = True, inferSchema = True)
		df = df.drop("Species")

		flatPmmlTransformer = FlatPMMLTransformer(evaluator)

		pmml_df = flatPmmlTransformer.transform(df)
		self.assertEqual(150, pmml_df.count())
		self.assertEqual(4 + (1 + 3) + 1, len(pmml_df.columns))

		nestedPmmlTransformer = NestedPMMLTransformer(evaluator)

		pmml_df = nestedPmmlTransformer.transform(df)
		self.assertEqual(150, pmml_df.count())
		self.assertEqual(4 + (1) + 1, len(pmml_df.columns))
