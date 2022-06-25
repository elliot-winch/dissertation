import json
from types import SimpleNamespace

def json_file_to_obj(file_name):
    config_file = open(file_name)
    return json.loads(config_file.read(), object_hook=lambda d: SimpleNamespace(**d))

def obj_to_json_file(object, file_name):
    with open(file_name, "w") as file:
            json.dump(object.__dict__, file, indent=2, default=lambda o: getattr(o, '__dict__', str(o)))
