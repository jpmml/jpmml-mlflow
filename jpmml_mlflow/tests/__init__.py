from abc import abstractmethod, ABC
from pyspark.context import SparkContext
from pyspark.sql import SparkSession
from typing import List
from unittest import TestCase

import mlflow
import shutil
import tempfile

class MLflowTest(TestCase):

	def setUp(self):
		self._tracking_dir = tempfile.mkdtemp()
		mlflow.set_tracking_uri(f"file://{self._tracking_dir}")
		mlflow.set_experiment("test")

	def tearDown(self):
		mlflow.set_tracking_uri(None)
		shutil.rmtree(self._tracking_dir)

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
