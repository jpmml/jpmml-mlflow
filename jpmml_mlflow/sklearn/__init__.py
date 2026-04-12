from jpmml_mlflow.flavor import add_pmml_flavor
from mlflow.models.signature import ModelSignature
from sklearn.base import BaseEstimator
from sklearn2pmml import make_pmml_pipeline, sklearn2pmml
from sklearn2pmml.pipeline import PMMLPipeline
from typing import Optional

import mlflow.sklearn

import logging
import numpy
import os
import sys
import tempfile

_logger = logging.getLogger(__name__)

def enhance_model(obj, signature: ModelSignature):
	active_fields = numpy.array(signature.inputs.input_names())
	target_fields = numpy.array(signature.outputs.input_names())

	if isinstance(obj, PMMLPipeline):
		obj.active_fields = active_fields
		obj.target_fields = target_fields
		_logger.warning("Replaced PMMLPipeline field names with signature field names")
	elif isinstance(obj, BaseEstimator):
		obj = make_pmml_pipeline(obj, active_fields = active_fields, target_fields = target_fields)
	else:
		_logger.warning("Failed to enhance model artifact with signature")

	return obj

def convert_model(obj, signature: Optional[ModelSignature] = None, input_example = None, precision = 1e-13, zeroThreshold = 1e-13) -> Optional[str]:

	if signature is not None:
		obj = enhance_model(obj, signature)

	if (input_example is not None) and isinstance(obj, PMMLPipeline):
		obj.verify(input_example, precision = precision, zeroThreshold = zeroThreshold)

	fd, pmml_path = tempfile.mkstemp(suffix = ".pmml")
	os.close(fd)

	try:
		sklearn2pmml(obj, pmml_path)
		return pmml_path
	except:
		_logger.warning("Failed to convert model artifact to PMML", exc_info = True)
		os.unlink(pmml_path)
		return None

log_model, save_model, load_model = add_pmml_flavor(sys.modules[__name__], mlflow.sklearn, "sk_model", convert_model)
