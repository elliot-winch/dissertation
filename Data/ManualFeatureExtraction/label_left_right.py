import cv2
import numpy as np

import argparse
from os import listdir, makedirs
from os.path import isfile, join, dirname, exists

from copy import deepcopy
from types import SimpleNamespace
import handle_json

def mean_sqaure_error(imageA, imageB):
    error = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    error /= float(imageA.shape[0] * imageA.shape[1])
    return error

def label_image(image, templates):
    errors = [mean_sqaure_error(image, template.image) for template in templates]
    print(errors) #debugging
    index_min = np.argmin(errors)
    return templates[index_min].label

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_folder_name", help="path to read images")
    parser.add_argument("-t", "--templates_file_name", help="path to templates json file")
    parser.add_argument("-o", "--output_file_name", help="path to output json labe file")
    args = parser.parse_args()

    #Load images
    png_file_names = [f for f in listdir(args.image_folder_name) if isfile(join(args.image_folder_name, f)) and f.endswith('.png')]

    #load template images
    templates = handle_json.json_file_to_obj(args.templates_file_name)

    #Create labels object
    labels = SimpleNamespace()
    labels.templates = templates.templates
    labels.labels = []

    #Load template images
    new_templates = deepcopy(templates)
    for template in new_templates.templates:
        template.image = cv2.imread(template.file_name)

    #Determine labels
    for i in range(len(png_file_names)):
        image = cv2.imread(join(args.image_folder_name, png_file_names[i]))
        labels.labels.append({
            "file_name" : png_file_names[i],
            "label" : label_image(image, new_templates.templates)
        })

    #Save labels
    handle_json.obj_to_json_file(labels, args.output_file_name)
