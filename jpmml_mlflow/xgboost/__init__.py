from jpmml_mlflow.flavor import add_pmml_flavor

import jpmml_mlflow.sklearn
import mlflow.xgboost

import sys

convert_model = jpmml_mlflow.sklearn.convert_model

log_model, save_model, load_model = add_pmml_flavor(sys.modules[__name__], mlflow.xgboost, "xgb_model", convert_model)
