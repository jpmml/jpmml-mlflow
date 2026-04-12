from jpmml_mlflow.flavor import add_pmml_flavor
from mlflow.models.signature import ModelSignature
from pandas import DataFrame
from typing import Optional
from xgboost import Booster

import jpmml_mlflow.sklearn
import mlflow.xgboost

import sys

def convert_model(xgb_model, signature: Optional[ModelSignature] = None, input_example = None, fmap = None) -> Optional[str]:
	if isinstance(xgb_model, Booster) and isinstance(fmap, DataFrame):
		xgb_model.fmap = fmap

	return jpmml_mlflow.sklearn.convert_model(xgb_model, signature = signature, input_example = input_example)

log_model, save_model, load_model = add_pmml_flavor(sys.modules[__name__], mlflow.xgboost, "xgb_model", convert_model, ("fmap", ))
