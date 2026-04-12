# 0.3.0 #

## Breaking changes

None.

## New features

* Added support for MLflow model signatures.

When a `signature` argument is passed to `log_model` or `save_model` functions, then the PMML converter uses it for customizing PMML model active and target field names.

* Added support for MLflow model example inputs.

When an `input_example` argument is passed to `log_model` or `save_model` functions, then the PMML converter uses it for generating a PMML model verification element.

For example, hardening a SkLearn pipeline:

```python
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import jpmml_mlflow.sklearn
import mlflow

X, y = ...
# SkLearn stores feature names, but not label name(s)
signature = infer_signature(X, y)
input_example = X.sample(n = ...)

pipeline = Pipeline(...)
pipeline.fit(X, y)

with mlflow.start_run() as run:
	jpmml_mlflow.sklearn.log_model(pipeline, name = "model", signature = signature, input_example = input_example)
```

## Minor improvements and fixes

* Updated JPMML-SparkML to latest.


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
