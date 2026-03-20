from jpmml_mlflow import evaluator
from jpmml_mlflow.util import load_classpath
from unittest import TestCase

import os

class UtilTest(TestCase):

	def test_classpath(self):
		jars = load_classpath(os.path.dirname(evaluator.__file__))
		self.assertEqual(15, len(jars))
		for jar in jars:
			self.assertTrue(jar.endswith(".jar"))

		jars = load_classpath(os.path.dirname(evaluator.tests.__file__))
		self.assertEqual(0, len(jars))