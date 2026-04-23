# 0.4.0 #

## Breaking changes

* Removed `jpmml_mlflow.evaluator` module.

The core functionality of this module was merged into the `jpmml_mlflow.evaluator_spark` module.

* Changed the return type of the `jpmml_mlflow.evaluator_spark.load_model()` function from JPMML-Evaluator object to PySpark transformer.

Previously, loading the JPMML-Evaluator object and wrapping it into a JPMML-Evaluaor-PySpark transformer were two separate steps:

```python
from jpmml_evaluator_pyspark import FlatPMMLTransformer

import jpmml_mlflow.evaluator_spark

evaluator = jpmml_mlflow.evaluator_spark.load_model(model_uri = ...)
pmml_transformer = FlatPMMLTransformer(evaluator)
```

The same is an atomic operation now:

```python
import jpmml_mlflow.evaluator_spark

pmml_transformer = jpmml_mlflow.evaluator_spark.load_model(model_uri = ...)
```

The default transformer type is `jpmml_evaluator_pyspark.FlatPMMLTransformer`.
Use the `transformer_type` parameter to specify a different transformer type.

* Removed the `classpath()` function from `jpmml_mlflow.spark` and `jpmml_mlflow.evaluator_spark` modules.

Use the `spark_jars()` function instead.

## New features

* Added `spark_jars(version: str)` function to modules that require PySpark classpath customization.

* Added `spark_jars_packages(version: str)` function to modules that require PySpark classpath customization.

## Minor improvements and fixes

* Updated PySpark2PMML to 0.11.0 or newer.

* Updated JPMML-Evaluator-PySpark to 0.3.0 or newer.

* Reduced the installed package size thousandfold (from 18 MB to 18 kB) by delegating JAR file management to `pyspark2pmml` and `jpmml-evaluator-pyspark` packages. 


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
