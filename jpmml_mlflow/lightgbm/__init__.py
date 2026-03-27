from jpmml_mlflow.flavor import add_pmml_flavor

import jpmml_mlflow.sklearn
import mlflow.lightgbm

import sys

convert_model = jpmml_mlflow.sklearn.convert_model

log_model, save_model, load_model = add_pmml_flavor(sys.modules[__name__], mlflow.lightgbm, "lgb_model", convert_model)
