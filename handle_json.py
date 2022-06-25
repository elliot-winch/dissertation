import json
from types import SimpleNamespace

from os import listdir
from os.path import isfile, join

json_extension = ".json"

def json_file_to_obj(file_name):
    config_file = open(file_name)
    return json.loads(config_file.read(), object_hook=lambda d: SimpleNamespace(**d))

def obj_to_json_file(object, file_name):
    with open(file_name, "w") as file:
            json.dump(object.__dict__, file, indent=2, default=lambda o: getattr(o, '__dict__', str(o)))

def load_jsons(folder):
    json_file_names = [f for f in listdir(folder) if isfile(join(folder, f)) and f.endswith(json_extension)]

    objs = []
    for json_file in json_file_names:
        objs.append(json_file_to_obj(join(folder, json_file)))

    return objs, json_file_names
