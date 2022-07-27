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
import argparse
import handle_json

uses = ["train", "val", "test"]
classes = ["Failure", "Success"]

def get_result_from_file_name(file_name):
    #remove file type then get result
    return file_name.split('.')[0].split('_')[4]

def arrange_files(config):
    sort_files(config.data_dir, config.sorted_data_dir, config.train_prop, config.val_prop, seed=config.seed)

def sort_files(input_folder, output_folder, train_prop, val_prop, seed=None):

    #If required, make this parameterisable
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

    if seed is not None:
        random.seed(seed)

    file_names = [f for f in listdir(input_folder) if isfile(join(input_folder, f))]
    random.shuffle(file_names)

    for i in range(len(file_names)):
        file_name = file_names[i]
        result = get_result_from_file_name(file_name)

        if result not in classes:
            continue

        #Split data set into train / test / validation
        usage = ""
        prop = i / len(file_names)
        if prop < train_prop:
            usage = uses[0]
        elif prop < train_prop + val_prop:
            usage = uses[1]
        else:
            usage = uses[2]

        location = output_folder + '/' + usage + '/' + result

        files[usage][result].append(location + '/' + file_name)
        shutil.copyfile(input_folder + '/' + file_name, location + '/' + file_name)

    return files

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folder", help="path to dataset")
    parser.add_argument("-o", "--output_folder", help="path to output folder")
    parser.add_argument("-t", "--train_prop", help="proportion of files to put in the train folder")
    parser.add_argument("-v", "--val_prop", help="proportion of files to put in the val folder")
    args = parser.parse_args()

    sort_files(args.input_folder, args.output_folder, float(args.train_prop), float(args.val_prop))
