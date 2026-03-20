import os

def _load_resource(name):
	resource_path = os.path.join(os.path.dirname(__file__), f"resources/{name}")
	with open(resource_path, "rb") as resource_file:
		return resource_file.read()
