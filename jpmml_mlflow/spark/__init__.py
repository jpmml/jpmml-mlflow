from jpmml_mlflow.spark import spark34, spark35, spark40, spark41
from jpmml_mlflow.util import load_classpath
from mlflow.models import Model
from mlflow.models.model import ModelInfo
from mlflow.models.signature import ModelSignature
from types import ModuleType
from typing import List, Optional

import jpmml_mlflow.pmml
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
	return load_classpath(os.path.dirname(spark_module.__file__))

def _convert(spark_model, input_example) -> Optional[str]:
	fd, pmml_path = tempfile.mkstemp(suffix = ".pmml")
	os.close(fd)

	try:
		from pyspark2pmml import PMMLBuilder

		PMMLBuilder(input_example.schema, spark_model) \
			.buildFile(pmml_path)
		return pmml_path
	except:
		_logger.warning("Failed to convert PySpark object to PMML", exc_info = True)
		os.unlink(pmml_path)
		return None

def log_model(spark_model, artifact_path = None, registered_model_name = None, name = None, **kwargs) -> ModelInfo:
	spark_flavor = sys.modules[__name__]
	return Model.log(artifact_path = name or artifact_path, flavor = spark_flavor, registered_model_name = registered_model_name, spark_model = spark_model, **kwargs)

def save_model(spark_model, path, mlflow_model: Optional[Model] = None, input_example = None, **kwargs) -> None:
	if input_example is not None:
		pmml_path = _convert(spark_model, input_example)
	else:
		_logger.warning("Skipping PMML flavor due to missing input_example")
		pmml_path = None

	if hasattr(input_example, "toPandas"):
		input_example = input_example.toPandas()

	if mlflow_model is None:
		mlflow_model = Model()

	mlflow.spark.save_model(spark_model, path = path, mlflow_model = mlflow_model, input_example = input_example, **kwargs)
	if pmml_path is not None:
		jpmml_mlflow.pmml.save_model(pmml_path, path = path, mlflow_model = mlflow_model)

load_model = mlflow.spark.load_model