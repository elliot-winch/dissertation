"""
Put files into folders based on Result

File Name Convention:
ID, Gender, Age, Surgery Type, Result, (File Type)
"""

import os
import os.path as path
from os import listdir
from os.path import isfile, join

data_dir = "Data"
new_data_dir = 'DividedData'

file_names = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]

for file_name in file_names:
    #From Convention - remove file type then get result
    result = file_name.split('.')[0].split('_')[4]
    location = new_data_dir + '/' + result

    if path.isdir(location) is False:
        os.mkdir(location)

    os.rename(data_dir + '/' + file_name, location + '/' + file_name)
