import os

def _find_resource(name):
	return os.path.join(os.path.dirname(__file__), f"resources/{name}")

def _load_resource(name):
	resource_path = _find_resource(name)
	with open(resource_path, "rb") as resource_file:
		return resource_file.read()
