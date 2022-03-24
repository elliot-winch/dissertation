"""
Uses file names to create an annotation csv

File Name Convention:
ID, Gender, Age, Surgery Type, Result, (File Type)
"""

from os import listdir
from os.path import isfile, join

data_dir = "Data"
csv_name = "Attributes"
header = "ID,Gender,Age,Surgery Type,Result"

file_names = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]
csv_content = [f.split('.')[0].replace('_', ',') for f in file_names]
#Do we need a header? csv_content.insert(0, header)
csv_content = '\n'.join(csv_content)

f = open(csv_name + ".csv", "w")
f.write(csv_content)
f.close()
