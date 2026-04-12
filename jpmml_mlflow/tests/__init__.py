from abc import abstractmethod, ABC
from lxml import etree
from mlflow.artifacts import download_artifacts
from mlflow.models import Model
from pyspark.context import SparkContext
from pyspark.sql import SparkSession
from typing import List
from unittest import TestCase

import jpmml_mlflow.pmml
import mlflow
import os
import pandas
import shutil
import tempfile

def _find_resource(name):
	return os.path.join(os.path.dirname(__file__), f"resources/{name}")

def _load_resource(name):
	resource_path = _find_resource(name)
	with open(resource_path, "rb") as resource_file:
		return resource_file.read()

def _load_iris():
	df = pandas.read_csv(_find_resource("Iris.csv"))

	iris_X = df[df.columns[0:4]]
	iris_y = df["Species"]

	return (iris_X, iris_y)

class MLflowTest(TestCase):

	def setUp(self):
		self._tracking_dir = tempfile.mkdtemp()
		mlflow.set_tracking_uri(f"file://{self._tracking_dir}")
		mlflow.set_experiment("test")

	def tearDown(self):
		mlflow.set_tracking_uri(None)
		shutil.rmtree(self._tracking_dir)

	def assertFlavors(self, run, expected_flavors):
		model_path = download_artifacts(f"runs:/{run.info.run_id}/model")
		mlflow_model = Model.load(model_path)
		for expected_flavor in expected_flavors:
			self.assertIn(expected_flavor, mlflow_model.flavors)

	def assertPMML(self, run, true_xpaths = [], false_xpaths = []):
		pmml_bytes = jpmml_mlflow.pmml.load_model(f"runs:/{run.info.run_id}/model")
		pmml_root = etree.fromstring(pmml_bytes)
		nsmap = {
			"pmml" : "http://www.dmg.org/PMML-4_4"
		}
		for true_xpath in true_xpaths:
			self.assertIsNotNone(pmml_root.find(true_xpath, nsmap), "Expected XPath '{}' not found in PMML".format(true_xpath))
		for false_xpath in false_xpaths:
			self.assertIsNone(pmml_root.find(false_xpath, nsmap), "Unexpected XPath '{}' found in PMML".format(false_xpath))

	def assertSignature(self, run, required_fields = [], prohibited_fields = []):
		true_xpaths = [".//pmml:DataField[@name='{}']".format(f) for f in required_fields]
		false_xpaths = [".//pmml:DataField[@name='{}']".format(f) for f in prohibited_fields]
		self.assertPMML(run, true_xpaths = true_xpaths, false_xpaths = false_xpaths)

	def assertInputExample(self, run, fields, n_samples = 10):
		true_xpaths = [".//pmml:ModelVerification/pmml:VerificationFields/pmml:VerificationField[@field='{}']".format(field) for field in fields]
		true_xpaths.append(".//pmml:ModelVerification/pmml:InlineTable/pmml:row[{}]".format(n_samples))
		self.assertPMML(run, true_xpaths = true_xpaths)

class PySparkTest(MLflowTest, ABC):

	@classmethod
	@abstractmethod
	def _spark_jars(cls) -> List[str]:
		raise NotImplementedError()

	@classmethod
	def setUpClass(cls):
		cls._spark = SparkSession.builder \
			.master("local") \
			.config("spark.jars", ",".join(cls._spark_jars())) \
			.getOrCreate()

	@classmethod
	def tearDownClass(cls):
		cls._spark.stop()
		# Shut down the underlying JVM, in order to avoid classpath conflicts between test classes
		if SparkContext._gateway:
			SparkContext._gateway.shutdown()
			SparkContext._gateway = None
			SparkContext._jvm = None
