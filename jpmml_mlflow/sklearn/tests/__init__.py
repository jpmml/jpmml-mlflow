from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

def _make_sk_model():
	iris = load_iris(as_frame = True)
	iris_X = iris.data
	iris_y = iris.target

	model = DecisionTreeClassifier(max_depth = 3, random_state = 42)
	model.fit(iris_X, iris_y)

	return model
