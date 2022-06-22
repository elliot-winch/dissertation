import handle_json

from os import listdir
from os.path import isfile, join

json_extension = ".json"

def load_results(resultsFolder):
    json_file_names = [f for f in listdir(resultsFolder) if isfile(join(resultsFolder, f)) and f.endswith(json_extension)]

    results = []
    for json_file in json_file_names:
        results.append(handle_json.json_file_to_obj(join(resultsFolder, json_file)))

    return results, json_file_names
