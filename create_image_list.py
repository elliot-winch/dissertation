import random
import math
import os
from os import listdir
from os.path import isfile, join

def write(lines, file_name):
    f = open(file_name, "w")
    f.write('\n'.join(lines))
    f.close()

seed = 1

location = "/scratch/Teaching/ew01000/retina_data"
data_dir = "retina_data"
labels_name = "labels.txt"
train_name = "train.txt"
val_name = "val.txt"
test_name = "test.txt"

train_perc = 0.90
val_perc = 0.09
#test_perc = 1 - train_perc - val_perc

test_content = []
val_content = []
train_content = []

dir_names = [f.name for f in os.scandir(data_dir) if f.is_dir()]

for i in range(len(dir_names)):
    class_name = dir_names[i]
    class_dir = data_dir + '/' + class_name
    file_names = [f for f in listdir(class_dir) if isfile(join(class_dir, f))]

    total_train = math.floor(len(file_names) * train_perc)
    total_val = math.floor(total_train + len(file_names) * val_perc)

    print("For class " + class_name + ": total_train is " + str(total_train) + " and total_val is " + str(total_val) + " and total test is " + str(len(file_names) - total_val))

    for j in range(len(file_names)):
        line = [location + '/' + class_name + '/' + file_names[j] + ' ' + str(i)]

        if j < total_train:
            train_content += line
        elif j < total_val:
            val_content += line
        else:
            test_content += line

random.seed(seed)
random.shuffle(train_content)
random.shuffle(val_content)
random.shuffle(test_content)

print("Total lines: test " + str(len(train_content)) + " train " + str(len(val_content)) + " test " + str(len(test_content)))

write(dir_names, labels_name)
write(train_content, train_name)
write(val_content, val_name)
write(test_content, test_name)
