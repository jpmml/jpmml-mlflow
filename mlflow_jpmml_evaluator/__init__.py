from py4j.java_gateway import JavaObject, JVMView

import glob
import mlflow_pmml
import os

def _load_classpath():
	resources_dir_path = os.path.join(os.path.dirname(__file__), "resources")
	return glob.glob(os.path.join(resources_dir_path, "*.jar"))

log_model = mlflow_pmml.log_model

save_model = mlflow_pmml.save_model

def load_model(model_uri, jvm: JVMView) -> JavaObject:
	pmml_bytes = mlflow_pmml.load_model(model_uri)

	pmmlIs = jvm.java.io.ByteArrayInputStream(pmml_bytes)
	try:
		return jvm.org.jpmml.evaluator.LoadingModelEvaluatorBuilder() \
			.load(pmmlIs) \
			.build()
	finally:
		pmmlIs.close()
