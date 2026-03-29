JPMML-MLflow [![Build Status](https://github.com/jpmml/jpmml-mlflow/workflows/pytest/badge.svg)](https://github.com/jpmml/jpmml-mlflow/actions?query=workflow%3A%22pytest%22)
============

PMML model flavors for MLflow.

# Motivation #

Every MLflow model artifact has at least one flavor.
The current flavors (eg. pickle, cloudpickle, ONNX) are optimized for **machine execution**.

Adding a PMML flavor to the default flavor(s) opens up models to **humans** and **AI agents** natively, enabling **higher-order tasks** such as model governance and AI-assisted data science.

Toggle the PMML flavor "on" or "off" with a one-line code change.

# Features #

JPMML-MLflow is the bridge between [Java PMML API](https://github.com/jpmml) and [MLflow](https://mlflow.org/).

The Predictive Model Markup Language (PMML) is an XML-based industry standard for representing statistical and data mining models.
PMML is first and foremost **applicable to structured data** workflows.
Unstructured data workflows are much better served by other more DNN-oriented standards such as ONNX, PyTorch TorchScript or TensorFlow SavedModel.

PMML is a single-file text-based representation, which makes it suitable for direct human and LLM consumption:

* Interpretation and auditing. See the model inner structure at various abstraction levels.
* Diffing. Perform structural and semantic comparisons between model versions.
* AI data science insights. Ask your favourite LLM to review the model, and suggest workflow changes to steer it in desired directions (eg. more predictive, explainable, or resource efficient).

PMML is equally suitable for execution.
In fact, the [JPMML-Evaluator](https://github.com/jpmml/jpmml-evaluator) library has high-quality and high-performance integrations available for all major ML platforms such as Python, R and Apache Spark (both Scala and PySpark variants).

# Prerequisites #

* MLflow 2.X or 3.X.
* PySpark 3.0 through 3.5, 4.0 or 4.1.
* Python 3.8 or newer.

# Installation #

Installing a release version from PyPI:

```bash
pip install jpmml-mlflow
```

Alternatively, installing the latest snapshot version from GitHub:

```
pip install --upgrade git+https://github.com/jpmml/jpmml-mlflow.git
```

The installed package contains all PMML flavor modules.

Some PMML flavor modules require third-party packages such as Scikit-Learn and PySpark.
JPMML-MLflow tries to preserve and use whatever versions are already available in the environment.

Installing into a new environment:

```bash
# Conversion side flavors
pip install jpmml-mlflow[sklearn,lightgbm,spark,xgboost]
# Evaluation side flavors
pip install jpmml-mlflow[evaluator,evaluator-spark]
```

# Usage #

Summary of PMML flavor modules:

* Foundation:
  * `jpmml_mlflow.pmml`. Saves and loads PMML XML documents as UTF-8 byte arrays (blobs).
* Conversion side:
  * `jpmml_mlflow.lightgbm`. Extends `mlflow.lightgbm` with PMML save when logging or saving a LightGBM artifact.
  * `jpmml_mlflow.sklearn`. Extends `mlflow.sklearn` with PMML save when logging or saving a Scikit-Learn artifact.
  * `jpmml_mlflow.spark`. Extends `mlflow.spark` with PMML save when logging or saving a PySpark artifact.
  * `jpmml_mlflow.xgboost`. Extends `mlflow.xgboost` with PMML save when logging or saving an XGBoost artifact.
* Evaluation side:
  * `jpmml_mlflow.evaluator`. Loads PMML into a JPMML-Evaluator artifact.
  * `jpmml_mlflow.evaluator-spark`. Extends `jpmml_mlflow.evaluator` with PySpark transformer support when loading a JPMML-Evaluator artifact.

## Foundation

### `jpmml_mlflow.pmml`

Low-level PMML save and load functionality, suitable for mixing into any existing MLflow workflow.

Default workflow:

```python
import mlflow
import mlflow.sklearn

with mlflow.start_run():
	sk_model = ...

	mlflow.sklearn.log_model(sk_model, name = "model")
```

PMML flavored workflow:

1. Create a `mlflow.models.Model` object and a workspace dir.
2. Populate the default state using a MLflow `save_model` function call.
3. Populate the PMML flavor state using the JPMML-MLflow `save_model` function call.
4. Log the workspace dir using the MLflow `log_artifacts` function call.

```python
from mlflow.models import Model

import jpmml_mlflow.pmml
import mlflow
import mlflow.sklearn
import tempfile

with mlflow.start_run():
	sk_model = ...

	mlflow_model = Model()
	workspace_dir = tempfile.mkdtemp()

	# Default Scikit-Learn state
	mlflow.sklearn.save_model(sk_model, path = workspace_dir, mlflow_model = mlflow_model)
	# Convert Scikit-Learn to PMML by whatever means
	pmml_path = convert_model(sk_model)
	# PMML flavor state
	jpmml_mlflow.pmml.save_model(pmml_path, path = workspace_dir, mlflow_model = mlflow_model)

	mlflow.log_artifacts(workspace_dir, artifact_path = "model")
```

Inspecting logged models for available flavors:

```python
from mlflow.artifacts import download_artifacts
from mlflow.models import Model

mlflow_model_path = download_artifacts(f"runs:/{run_id}/model")

mlflow_model = Model.load(mlflow_model_path)
print(mlflow_model.flavors)

pmml_flavor = mlflow_model.flavors["pmml"]
print(pmml_flavor)
```

Loading the PMML representation of the logged model:

```python
import jpmml_mlflow.pmml

pmml_bytes = jpmml_mlflow.pmml.load_model(f"runs:/{run_id}/model")
```

## Conversion side

### `jpmml_mlflow.lightgbm`

A PMML-aware replacement for the `mlflow.lightgbm` module.

```python
import jpmml_mlflow.lightgbm
import mlflow
#import mlflow.lightgbm

with mlflow.start_run():
	jpmml_mlflow.lightgbm.log_model(lgb_model, name = "model")
```

The LightGBM artifact can be either a `lightgbm.Booster` or a Scikit-Learn wrapper class.

### `jpmml_mlflow.sklearn`

A PMML-aware replacement for the `mlflow.sklearn` module.

```python
import jpmml_mlflow.sklearn
import mlflow
#import mlflow.sklearn

with mlflow.start_run():
	jpmml_mlflow.sklearn.log_model(sk_model, name = "model")
```

The conversion task is delegated to the [`sklearn2pmml`](https://github.com/jpmml/sklearn2pmml) package.

The list of supported transformer and estimator types is available [here](https://github.com/jpmml/jpmml-sklearn/blob/master/features.md).
Develop and register custom [JPMML-SkLearn](https://github.com/jpmml/jpmml-sklearn) converter classes as necessary.

### `jpmml_mlflow.spark`

A PMML-aware replacement for the `mlflow.spark` module.

```python
import jpmml_mlflow.spark
import mlflow
#import mlflow.spark

with mlflow.start_run():
	# The input_example must be a training DataFrame excerpt
	jpmml_mlflow.spark.log_model(spark_model, name = "model", input_example = df.limit(10))
```

The conversion task is delegated to the [`pyspark2pmml`](https://github.com/jpmml/pyspark2pmml) package.

The list of supported transformer and estimator types is available [here](https://github.com/jpmml/jpmml-sparkml/blob/master/features.md).
Develop and register custom [JPMML-SparkML](https://github.com/jpmml/jpmml-sparkml) converter classes as necessary.

PySpark2PMML requires [JPMML-SparkML](https://github.com/jpmml/jpmml-sparkml) library JAR files available on PySpark classpath.
They are typically added using `--jars` or `--packages` command-line options.

JPMML-MLflow incorporates JPMML dependencies for all supported PySpark versions.
Access the right JAR fileset using the `jpmml_mlflow.spark.classpath(version: str)` utility function.

Configuring the PySpark classpath programmatically:

```python
from pyspark.sql import SparkSession

import jpmml_mlflow.spark
import pyspark

spark_jars = jpmml_mlflow.spark.classpath(version = pyspark.__version__)

spark = SparkSession.builder \
	.config("spark.jars", ",".join(spark_jars)) \
	.getOrCreate()
```

### `jpmml_mlflow.xgboost`

A PMML-aware replacement for the `mlflow.xgboost` module.

```python
import jpmml_mlflow.xgboost
import mlflow
#import mlflow.xgboost

with mlflow.start_run():
	jpmml_mlflow.xgboost.log_model(xgb_model, name = "model")
```

The XGBoost artifact can be either a `xgboost.Booster` or a Scikit-Learn wrapper class.

## Evaluation side

### `jpmml_mlflow.evaluator_spark`

Apache Spark is poorly interoperable with Python artifacts.
The default approach is to turn the Python function flavor of a model artifact into a Python UDF using the `mlflow.pyfunc.spark_udf()` utility function.
However, scoring Spark DataFrame data with Python UDF is a high friction operation, because the data needs to be re-serialized to pass it from one environment to another.

The `jpmml_mlflow.evaluator-spark` module provides a compelling alternative to that.
The PMML flavor of a model artifact is loaded into a JPMML-Evaluator artifact, which is then wrapped into an Apache Spark transformer.
This transformer scores Spark DataFrame data natively on executors via `mapPartitions`. There is no JVM process boundary, no data serialization overhead, let alone any cluster-side ML framework configuration or dependencies.

Loading a logged model:

```python
import jpmml_mlflow.evaluator_spark

evaluator = jpmml_mlflow.evaluator_spark.load_model(f"runs:/{run_id}/model")
```

Constructing a PySpark transformer, and scoring data:

```python
from jpmml_evaluator_pyspark import FlatPMMLTransformer, NestedPMMLTransformer

# Flat layout of result columns
pmml_transformer = FlatPMMLTransformer(evaluator)
# Nested layout of result columns
#pmml_transformer = NestedPMMLTransformer(evaluator)

#pmml_schema = pmml_transformer.transformSchema(df.schema)
#print(pmml_schema)

pmml_df = pmml_transformer.transform(df)
pmml_df.show()
```

The PySpark PMML transformers are provided by the [`jpmml-evaluator-pyspark`](https://github.com/jpmml/jpmml-evaluator-pyspark) package.

JPMML-Evaluator-PySpark requires [JPMML-Evaluator-Spark](https://github.com/jpmml/jpmml-evaluator-spark) and [JPMML-Evaluator](https://github.com/jpmml/jpmml-evaluator) library JAR files available on PySpark classpath.
They are typically added using `--jars` or `--packages` command-line options.

JPMML-MLflow incorporates JPMML dependencies for all supported PySpark versions.
Access the right JAR fileset using the `jpmml_mlflow.evaluator_spark.classpath(version: str)` utility function.

Configuring the PySpark classpath programmatically:

```python
from pyspark.sql import SparkSession

import jpmml_mlflow.evaluator_spark
import pyspark

spark_jars = jpmml_mlflow.evaluator_spark.classpath(version = pyspark.__version__)

spark = SparkSession.builder \
	.config("spark.jars", ",".join(spark_jars)) \
	.getOrCreate()
```

# License #

JPMML-MLflow is licensed under the terms and conditions of the [GNU Affero General Public License, Version 3.0](https://www.gnu.org/licenses/agpl-3.0.html).
For a quick summary of your rights ("Can") and obligations ("Cannot" and "Must") under AGPLv3, please refer to [TLDRLegal](https://tldrlegal.com/license/gnu-affero-general-public-license-v3-(agpl-3.0)).

If you would like to use JPMML-MLflow in a proprietary software project, then it is possible to enter into a licensing agreement which makes it available under the terms and conditions of the [BSD 3-Clause License](https://opensource.org/licenses/BSD-3-Clause) instead.

# Additional information #

JPMML-MLflow is developed and maintained by Openscoring Ltd, Estonia.

Interested in using JPMML software in your software? Please contact [info@openscoring.io](mailto:info@openscoring.io)
