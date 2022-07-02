import json
from types import SimpleNamespace

from os import listdir, makedirs
from os.path import isfile, join, dirname, exists
import pathlib

json_extension = ".json"

def json_file_to_obj(file_name):
    config_file = open(file_name)
    return json.loads(config_file.read(), object_hook=lambda d: SimpleNamespace(**d))

def obj_to_json_file(object, file_name):
    #Ensure folder exists before writing
    directory = pathlib.Path(dirname(file_name))
    if not exists(directory):
        makedirs(directory)

    with open(file_name, "w") as file:
            json.dump(object.__dict__, file, indent=2, default=lambda o: getattr(o, '__dict__', str(o)))


def load_jsons(folder, order_by=None):
    json_file_names = [f for f in listdir(folder) if isfile(join(folder, f)) and f.endswith(json_extension)]

    objs = []
    for json_file in json_file_names:
        objs.append(json_file_to_obj(join(folder, json_file)))

    if order_by is not None:
        indices = sorted(range(len(objs)), key=order_by(objs).__getitem__)
        objs[:] = [objs[i] for i in indices]
        json_file_names[:] = [json_file_names[i] for i in indices]

    return objs, json_file_names
