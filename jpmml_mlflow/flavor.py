from mlflow.models import Model
from mlflow.models.model import ModelInfo
from mlflow.models.signature import ModelSignature
from typing import Optional

import jpmml_mlflow.pmml

def add_pmml_flavor(module, mlflow_module, model_kwarg, convert_model, convert_model_kwargs = ()):

	def log_model(model, artifact_path = None, registered_model_name = None, name = None, **kwargs) -> ModelInfo:
		return Model.log(artifact_path = name or artifact_path, flavor = module, registered_model_name = registered_model_name, **{model_kwarg : model}, **kwargs)

	def save_model(model = None, path = None, mlflow_model: Optional[Model] = None, signature: Optional[ModelSignature] = None, **kwargs) -> None:
		model = model if model is not None else kwargs.pop(model_kwarg)

		pmml_path = convert_model(model, signature = signature, **{k : kwargs.pop(k) for k in convert_model_kwargs if k in kwargs})

		if mlflow_model is None:
			mlflow_model = Model()

		mlflow_module.save_model(model, path = path, mlflow_model = mlflow_model, signature = signature, **kwargs)
		if pmml_path is not None:
			jpmml_mlflow.pmml.save_model(pmml_path, path = path, mlflow_model = mlflow_model)

	load_model = mlflow_module.load_model

	return (log_model, save_model, load_model)
