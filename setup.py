from setuptools import setup, find_packages

setup(
	name = "jpmml-mlflow",
	version = "0.1.0",
	license = "GNU Affero General Public License (AGPL) version 3.0",
	packages = find_packages(exclude = ["*.tests.*", "*.tests"]),
	package_data = {
		"" : ["resources/*.jar"],
	},
	install_requires = [
		"mlflow>=2.0,<3.0",
		"py4j",
		"pyspark"
	]
)
