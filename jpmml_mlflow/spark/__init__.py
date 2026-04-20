from jpmml_mlflow.flavor import add_pmml_flavor
from mlflow.models import Model
from mlflow.models.signature import ModelSignature
from pyspark.sql import SparkSession
from pyspark2pmml import PMMLBuilder
from typing import Optional

import mlflow.spark

import logging
import os
import pyspark2pmml
import sys
import tempfile

_logger = logging.getLogger(__name__)

spark_jars = pyspark2pmml.spark_jars

def convert_model(spark_model, input_example_schema, signature: Optional[ModelSignature] = None, input_example = None) -> Optional[str]:
	fd, pmml_path = tempfile.mkstemp(suffix = ".pmml")
	os.close(fd)

	try:
		pmmlBuilder = PMMLBuilder(input_example_schema, spark_model)
		if input_example is not None:
			spark = SparkSession.getActiveSession()
			pmmlBuilder.verify(spark.createDataFrame(input_example))
		pmmlBuilder.buildFile(pmml_path)
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

