from jpmml_mlflow.flavor import add_pmml_flavor
from jpmml_mlflow.spark import shared, spark34, spark35, spark40, spark41
from jpmml_mlflow.util import load_classpath
from mlflow.models import Model
from types import ModuleType
from typing import List, Optional

import mlflow.spark

import logging
import os
import pyspark
import sys
import tempfile

_logger = logging.getLogger(__name__)

def _spark_module(version: str) -> ModuleType:
	if version.startswith("3.4."):
		return spark34
	elif version.startswith("3.5."):
		return spark35
	elif version.startswith("4.0."):
		return spark40
	elif version.startswith("4.1."):
		return spark41
	else:
		raise ValueError(f"Apache Spark version {version} is not supported")

def classpath(version: str = None) -> List[str]:
	if version is None:
		version = pyspark.__version__

	spark_module = _spark_module(version)
	spark_jars = load_classpath(os.path.dirname(spark_module.__file__))

	shared_jars = load_classpath(os.path.dirname(shared.__file__))

	return spark_jars + shared_jars

def convert_model(spark_model, input_example_schema) -> Optional[str]:
	fd, pmml_path = tempfile.mkstemp(suffix = ".pmml")
	os.close(fd)

	try:
		from pyspark2pmml import PMMLBuilder

		PMMLBuilder(input_example_schema, spark_model) \
			.buildFile(pmml_path)
		return pmml_path
	except:
		_logger.warning("Failed to convert PySpark object to PMML", exc_info = True)
		os.unlink(pmml_path)
		return None

log_model, _save_model, load_model = add_pmml_flavor(sys.modules[__name__], mlflow.spark, "spark_model", convert_model, ("input_example_schema", ))

def save_model(spark_model, path, mlflow_model: Optional[Model] = None, input_example = None, **kwargs) -> None:
	if input_example is None:
		_logger.warning("Skipping PMML flavor due to missing input_example")
		mlflow.spark.save_model(spark_model, path = path, mlflow_model = mlflow_model, input_example = input_example, **kwargs)
	else:
		input_example_schema = input_example.schema
		if hasattr(input_example, "toPandas"):
			input_example = input_example.toPandas()
		_save_model(spark_model, path = path, mlflow_model = mlflow_model, input_example = input_example, input_example_schema = input_example_schema, **kwargs)

