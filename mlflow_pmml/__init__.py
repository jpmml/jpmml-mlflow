from mlflow.artifacts import download_artifacts
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.utils.model_utils import _get_flavor_configuration
from typing import Union

import os

FLAVOR_NAME = "pmml"
FLAVOR_DATA_FILE = "model.pmml"

def save_model(pmml: Union[bytes, str], path, mlflow_model = None):
	if isinstance(pmml, bytes):
		pass
	elif isinstance(pmml, str):
		pmml = pmml.encode("utf-8")
	else:
		raise TypeError("PMML is not bytes or str")

	if mlflow_model is None:
		mlflow_model = Model()

	os.makedirs(path, exist_ok = True)
	pmml_path = os.path.join(path, FLAVOR_DATA_FILE)
	with open(pmml_path, "wb") as pmml_file:
		pmml_file.write(pmml)

	flavor_conf = {
		"data" : FLAVOR_DATA_FILE
	}
	mlflow_model.add_flavor(FLAVOR_NAME, **flavor_conf)

	model_path = os.path.join(path, MLMODEL_FILE_NAME)
	mlflow_model.save(model_path)

def load_model(model_uri):
	model_path = download_artifacts(model_uri)
	flavor_conf = _get_flavor_configuration(model_path, flavor_name = FLAVOR_NAME)

	pmml_path = os.path.join(model_path, flavor_conf["data"])
	with open(pmml_path, "rb") as pmml_file:
		return pmml_file.read()
