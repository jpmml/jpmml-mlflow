from jpmml_mlflow import evaluator
from jpmml_mlflow.evaluator_spark import spark3x, spark4x
from jpmml_mlflow.util import load_classpath
from py4j.java_gateway import JavaObject, JVMView
from pyspark.sql import SparkSession
from types import ModuleType
from typing import List

import os
import pyspark

def _spark_module(version: str) -> ModuleType:
	if version.startswith("3."):
		return spark3x
	elif version.startswith("4."):
		return spark4x
	else:
		raise ValueError(f"Apache Spark version {version} is not supported")

def classpath(version: str = None) -> List[str]:
	if version is None:
		version = pyspark.__version__

	spark_module = _spark_module(version)
	evaluator_spark_jars = load_classpath(os.path.dirname(spark_module.__file__))

	evaluator_jars = evaluator.classpath()

	# Current module's JARs first, parent module's JARs second
	return evaluator_spark_jars + evaluator_jars

log_model = evaluator.log_model

save_model = evaluator.save_model

def load_model(model_uri, jvm: JVMView = None, **kwargs) -> JavaObject:
	if jvm is None:
		spark = SparkSession.getActiveSession()
		jvm = spark._jvm

	return evaluator.load_model(model_uri, jvm = jvm, **kwargs)
