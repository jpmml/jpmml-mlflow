from jpmml_evaluator_pyspark import FlatPMMLTransformer, NestedPMMLTransformer, PMMLTransformer
from jpmml_mlflow import pmml
from py4j.java_gateway import JVMView
from pyspark.sql import SparkSession

import jpmml_evaluator_pyspark

spark_jars = jpmml_evaluator_pyspark.spark_jars

spark_jars_packages = jpmml_evaluator_pyspark.spark_jars_packages

log_model = pmml.log_model

save_model = pmml.save_model

def load_model(model_uri, transformer_type = FlatPMMLTransformer, jvm: JVMView = None) -> PMMLTransformer:
	if jvm is None:
		spark = SparkSession.getActiveSession()
		jvm = spark._jvm

	pmml_bytes = pmml.load_model(model_uri)

	pmmlIs = jvm.java.io.ByteArrayInputStream(pmml_bytes)
	try:
		evaluator = jvm.org.jpmml.evaluator.LoadingModelEvaluatorBuilder() \
			.load(pmmlIs) \
			.build()
	finally:
		pmmlIs.close()

	return transformer_type(evaluator)
