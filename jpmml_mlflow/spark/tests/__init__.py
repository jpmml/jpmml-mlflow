from jpmml_mlflow.tests import _find_resource
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import RFormula
from pyspark.sql.functions import rand

def _make_spark_model(spark):
	df = spark.read.csv(_find_resource("Iris.csv"), header = True, inferSchema = True)
	signature = None
	input_example = df.orderBy(rand(seed = 42)).limit(10)

	formula = RFormula(formula = "Species ~ .")
	classifier = DecisionTreeClassifier()
	pipeline = Pipeline(stages = [formula, classifier])
	pipeline_model = pipeline.fit(df)

	return (pipeline_model, signature, input_example)
