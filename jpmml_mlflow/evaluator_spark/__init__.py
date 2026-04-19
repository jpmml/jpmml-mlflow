from jpmml_evaluator_pyspark import _jars
from jpmml_mlflow import evaluator
from py4j.java_gateway import JavaObject, JVMView
from pyspark.sql import SparkSession
from typing import List

def classpath(version: str = None) -> List[str]:
	return _jars(version = version)

log_model = evaluator.log_model

save_model = evaluator.save_model

def load_model(model_uri, jvm: JVMView = None, **kwargs) -> JavaObject:
	if jvm is None:
		spark = SparkSession.getActiveSession()
		jvm = spark._jvm

	return evaluator.load_model(model_uri, jvm = jvm, **kwargs)
