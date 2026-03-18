from mlflow_pmml import save_model, load_model
from tempfile import TemporaryDirectory
from unittest import TestCase

PMML_BYTES = b"<PMML/>"

class PMMLTest(TestCase):

	def test_bytes(self):
		with TemporaryDirectory() as path:
			save_model(pmml = PMML_BYTES, path = path)
			pmml_bytes = load_model(path)
		self.assertEqual(PMML_BYTES, pmml_bytes)

	def test_str(self):
		pmml_str = PMML_BYTES.decode("utf-8")
		with TemporaryDirectory() as path:
			save_model(pmml = pmml_str, path = path)
			pmml_bytes = load_model(path)
		self.assertEqual(PMML_BYTES, pmml_bytes)
