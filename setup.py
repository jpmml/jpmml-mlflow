from setuptools import setup, find_packages

exec(open("jpmml_mlflow/metadata.py").read())

setup(
	name = "jpmml-mlflow",
	version = __version__,
	description = "PMML model flavors for MLflow",
	author = "Villu Ruusmann",
	author_email = "villu.ruusmann@gmail.com",
	url = "https://github.com/jpmml/jpmml-mlflow",
	download_url = "https://github.com/jpmml/jpmml-mlflow/archive/" + __version__ + ".tar.gz",
	license = __license__,
	classifiers = [
		"Development Status :: 5 - Production/Stable",
		"Operating System :: OS Independent",
		"Programming Language :: Python",
		"Intended Audience :: Developers",
		"Intended Audience :: Science/Research",
		"Topic :: Software Development",
		"Topic :: Scientific/Engineering"
	],
	packages = find_packages(exclude = ["*.tests.*", "*.tests"]),
	package_data = {
		"" : ["resources/*.jar"],
	},
	exclude_package_data = {
		"" : ["README.md"],
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
		"lightgbm" : [
			"lightgbm",
			"sklearn2pmml",
		],
		"sklearn" : [
			"scikit-learn",
			"sklearn2pmml",
		],
		"spark" : [
			"pyspark>=3.0",
			"pyspark2pmml>=0.10.0",
		],
		"xgboost" : [
			"xgboost",
			"sklearn2pmml",
		]
	},
)
