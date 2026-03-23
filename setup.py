from setuptools import setup, find_packages

exec(open("jpmml_mlflow/metadata.py").read())

setup(
	name = "jpmml-mlflow",
	version = __version__,
	author = "Villu Ruusmann",
	author_email = "villu.ruusmann@gmail.com",
	license = __license__,
	packages = find_packages(exclude = ["*.tests.*", "*.tests"]),
	package_data = {
		"" : ["resources/*.jar"],
	},
	python_requires = ">=3.8",
	install_requires = [
		"mlflow>=2.0",
	],
	extras_require = {
		"evaluator" : [
			"py4j",
		],
		"evaluator-spark" : [
			"jpmml-evaluator-pyspark",
			"py4j",
			"pyspark>=3.0",
		],
		"sklearn" : [
			"scikit-learn",
			"sklearn2pmml",
		],
		"spark" : [
			"pyspark>=3.0",
			"pyspark2pmml>=0.10.0",
		]
	},
)
