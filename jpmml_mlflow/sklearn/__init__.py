from jpmml_mlflow.flavor import add_pmml_flavor
from mlflow.models.signature import ModelSignature
from sklearn2pmml import sklearn2pmml
from typing import Optional

import mlflow.sklearn

import logging
import os
import sys
import tempfile

_logger = logging.getLogger(__name__)

def convert_model(obj, signature: Optional[ModelSignature] = None) -> Optional[str]:
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
