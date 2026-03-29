# 0.2.0 #

## Breaking changes

None.

## New features

* Added `jpmml_mlflow.lightgbm` module.

A drop-in replacement for the `mlflow.lightgbm` module.
To start logging LightGBM artifacts with a PMML flavor, simply replace the `mlflow.lightgbm` module with `jpmml_mlflow.lightgbm` (ie. add the "jpmml_" prefix to current module name).

Default:

```python
import mlflow
import mlflow.lightgbm

with mlflow.start_run():
	mlflow.lightgbm.log_model(lgb_model, name = "model")
```

The same, but with an extra PMML flavor:

```python
import jpmml_mlflow.lightgbm
import mlflow

with mlflow.start_run():
	jpmml_mlflow.lightgbm.log_model(lgb_model, name = "model")
```

* Added `jpmml_mlflow.xgboost` module.

A drop-in replacement for the `mlflow.xgboost` module.

## Minor improvements and fixes

* Added `convert_model(model)` functions to all JPMML-MLflow flavor modules.

JPMML-MLflow extends MLflow conventions by assuming that a flavor model provides a `convert_model` function in addition to `log_model`, `save_model` and `load_model` functions.

End users can call the `convert_model` function to verify that the PMML converter supports their model artifacts.

Checking if some Scikit-Learn artifact is fully supported:

```python
import jpmml_mlflow.sklearn

sk_model = ...

pmml_path = jpmml_mlflow.sklearn.convert_model(sk_model)
print(pmml_path)
```

* Added `jpmml_mlflow.flavor.add_pmml_flavor()` utility function.
