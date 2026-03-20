from jpmml_mlflow import pmml
from jpmml_mlflow.util import load_classpath
from py4j.java_gateway import JavaObject, JVMView

import os

def classpath():
	return load_classpath(os.path.dirname(__file__))

log_model = pmml.log_model

save_model = pmml.save_model

def load_model(model_uri, jvm: JVMView) -> JavaObject:
	pmml_bytes = pmml.load_model(model_uri)

	pmmlIs = jvm.java.io.ByteArrayInputStream(pmml_bytes)
	try:
		return jvm.org.jpmml.evaluator.LoadingModelEvaluatorBuilder() \
			.load(pmmlIs) \
			.build()
	finally:
		pmmlIs.close()
