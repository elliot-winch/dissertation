import json
from types import SimpleNamespace

def json_file_to_obj(file_name):
    config_file = open(file_name)
    return json.loads(config_file.read(), object_hook=lambda d: SimpleNamespace(**d))
