from unittest import TestCase

import mlflow
import os
import shutil
import tempfile

def _load_resource(name):
	resource_path = os.path.join(os.path.dirname(__file__), f"resources/{name}")
	with open(resource_path, "rb") as resource_file:
		return resource_file.read()

class MLFlowTest(TestCase):

	def setUp(self):
		self._tracking_dir = tempfile.mkdtemp()
		mlflow.set_tracking_uri(f"file://{self._tracking_dir}")
		mlflow.set_experiment("test")

	def tearDown(self):
		mlflow.set_tracking_uri(None)
		shutil.rmtree(self._tracking_dir)
