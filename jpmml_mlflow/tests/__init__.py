from unittest import TestCase

import mlflow
import shutil
import tempfile

class MLFlowTest(TestCase):

	def setUp(self):
		self._tracking_dir = tempfile.mkdtemp()
		mlflow.set_tracking_uri(f"file://{self._tracking_dir}")
		mlflow.set_experiment("test")

	def tearDown(self):
		mlflow.set_tracking_uri(None)
		shutil.rmtree(self._tracking_dir)
