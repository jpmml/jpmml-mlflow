from mlflow.models import Model
from mlflow.models.model import ModelInfo
from mlflow.models.signature import ModelSignature
from typing import Optional

import jpmml_mlflow.pmml
import jpmml_mlflow.sklearn
import mlflow.lightgbm

import sys

def _convert(lgb_model) -> Optional[str]:
	return jpmml_mlflow.sklearn._convert(lgb_model)

def log_model(lgb_model, artifact_path = None, registered_model_name = None, name = None, **kwargs) -> ModelInfo:
	lightgbm_flavor = sys.modules[__name__]
	return Model.log(artifact_path = name or artifact_path, flavor = lightgbm_flavor, registered_model_name = registered_model_name, lgb_model = lgb_model, **kwargs)

def save_model(lgb_model, path, mlflow_model: Optional[Model] = None, signature: Optional[ModelSignature] = None, **kwargs) -> None:
	pmml_path = _convert(lgb_model)

	if mlflow_model is None:
		mlflow_model = Model()

	mlflow.lightgbm.save_model(lgb_model, path = path, mlflow_model = mlflow_model, signature = signature, **kwargs)
	if pmml_path is not None:
		jpmml_mlflow.pmml.save_model(pmml_path, path = path, mlflow_model = mlflow_model)

load_model = mlflow.lightgbm.load_model
