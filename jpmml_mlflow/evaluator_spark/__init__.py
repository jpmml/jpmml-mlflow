from jpmml_mlflow import pmml
from py4j.java_gateway import JavaObject, JVMView
from pyspark.sql import SparkSession

import jpmml_evaluator_pyspark

spark_jars = jpmml_evaluator_pyspark.spark_jars

log_model = pmml.log_model

save_model = pmml.save_model

def load_model(model_uri, jvm: JVMView = None) -> JavaObject:
	if jvm is None:
		spark = SparkSession.getActiveSession()
		jvm = spark._jvm

	pmml_bytes = pmml.load_model(model_uri)

	pmmlIs = jvm.java.io.ByteArrayInputStream(pmml_bytes)
	try:
		return jvm.org.jpmml.evaluator.LoadingModelEvaluatorBuilder() \
			.load(pmmlIs) \
			.build()
	finally:
		pmmlIs.close()
