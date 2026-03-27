from mlflow.models import Model
from mlflow.models.model import ModelInfo
from mlflow.models.signature import ModelSignature
from typing import Optional

import jpmml_mlflow.pmml
import mlflow.sklearn

import logging
import os
import sys
import tempfile

_logger = logging.getLogger(__name__)

def _convert(obj) -> Optional[str]:
	fd, pmml_path = tempfile.mkstemp(suffix = ".pmml")
	os.close(fd)

	try:
		from sklearn2pmml import sklearn2pmml

		sklearn2pmml(obj, pmml_path)
		return pmml_path
	except:
		_logger.warning("Failed to convert model artifact to PMML", exc_info = True)
		os.unlink(pmml_path)
		return None

def log_model(sk_model, artifact_path = None, registered_model_name = None, name = None, **kwargs) -> ModelInfo:
	sklearn_flavor = sys.modules[__name__]
	return Model.log(artifact_path = name or artifact_path, flavor = sklearn_flavor, registered_model_name = registered_model_name, sk_model = sk_model, **kwargs)

def save_model(sk_model, path, mlflow_model: Optional[Model] = None, signature: Optional[ModelSignature] = None, **kwargs) -> None:
	pmml_path = _convert(sk_model)

	if mlflow_model is None:
		mlflow_model = Model()

	mlflow.sklearn.save_model(sk_model, path = path, mlflow_model = mlflow_model, signature = signature, **kwargs)
	if pmml_path is not None:
		jpmml_mlflow.pmml.save_model(pmml_path, path = path, mlflow_model = mlflow_model)

load_model = mlflow.sklearn.load_model