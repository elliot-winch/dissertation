"""
Put files into folders based on Usage and Result

File Name Convention:
ID, Gender, Age, Surgery Type, Result, (File Type)
"""

import os
from os import listdir
import os.path as path
from os.path import isfile, join
import random
import sys
import shutil


import handle_json

uses = ["train", "val", "test"]

files = {
    "train" :
    {
        "Failure" : [],
        "Success" : []
    },
    "val" :
    {
        "Failure" : [],
        "Success" : []
    },
    "test" :
    {
        "Failure" : [],
        "Success" : []
    },
}

def arrange_files(config):

    if os.path.exists(config.data_dir) is False:
        print("Error: No data folder found at " + config.data_dir)
        exit()

    if os.path.exists(config.sorted_data_dir) and os.path.isdir(config.sorted_data_dir):
        shutil.rmtree(config.sorted_data_dir)

    random.seed(config.seed)

    file_names = [f for f in listdir(config.data_dir) if isfile(join(config.data_dir, f))]
    random.shuffle(file_names)

    for i in range(len(file_names)):
        sort_file(config, file_names, i)

def sort_file(config, file_names, index):
    file_name = file_names[index]
    #From Convention - remove file type then get result
    result = file_name.split('.')[0].split('_')[4]

    if result not in config.classes:
        return

    #Split data set into train / test / validation
    usage = ""
    prop = index / len(file_names)
    if prop < config.train_prop:
        usage = uses[0]
    elif prop < config.train_prop + config.val_prop:
        usage = uses[1]
    else:
        usage = uses[2]

    location = config.sorted_data_dir + '/' + usage + '/' + result

    if path.isdir(location) is False:
        if path.isdir(config.sorted_data_dir + '/' + usage) is False:
            if path.isdir(config.sorted_data_dir) is False:
                os.mkdir(config.sorted_data_dir)
            os.mkdir(config.sorted_data_dir + '/' + usage)
        os.mkdir(location)

    files[usage][result].append(location + '/' + file_name)
    shutil.copyfile(config.data_dir + '/' + file_name, location + '/' + file_name)

if __name__ == '__main__':

    if len(sys.argv) > 1:
        config_file_name = sys.argv[1]
        arrange_files(handle_json.json_file_to_obj(config_file_name))
    else:
        print("Please provide config file path")
