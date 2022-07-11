import cv2
import numpy as np
from math import sqrt, pi

import argparse
from os import listdir, makedirs
from os.path import isfile, join, dirname, exists

from types import SimpleNamespace
import handle_json


image_center = [199.5, 199.5]

#property_image: greyscale image of single property (e.g. horseshoe tears)
def get_property_info(property_image, max_property_count=8):

    cv2.imshow('property_image', property_image)

    num_labels, component_labeled_image, stats, centroids = cv2.connectedComponentsWithStats(property_image)

    properties_info = []

    #Label 0 is the background - pixels not within a feature
    for i in range(1, min(num_labels, max_property_count)):
        component_image = cv2.inRange(component_labeled_image, i, i)

        property_info = SimpleNamespace()
        property_info.size = stats[i, cv2.CC_STAT_AREA]
        # can also access width and height from stats if needed

        relative_centroid = [centroids[i][0] - image_center[0], centroids[i][1] - image_center[1]]

        property_info.distance = sqrt (relative_centroid[0] * relative_centroid[0] + relative_centroid[1] * relative_centroid[1])
        #convert to clockwise by negating x coordinate
        angle = np.arctan2(relative_centroid[1], -relative_centroid[0])
        #with midnight being 0 radians
        property_info.rotation = (angle + (3 * pi / 2)) % (pi * 2)

        moments = cv2.moments(component_image)
        property_info.hu_moments = cv2.HuMoments(moments).tolist()

        print(type(property_info.hu_moments))
        properties_info.append(property_info)

        #TEMP debugging
        print("Property info {}: size {} distance {} angle {} rotation {} hu {}".format(i, property_info.size, property_info.distance, angle, property_info.rotation, property_info.hu_moments))
        cv2.imshow('Single property image {}'.format(i),component_image)

    return properties_info


def get_feature_vector(scan, property_dictionary):

    feature_vector = []

    print(len(property_dictionary.properties))

    for property in property_dictionary.properties:
        prop_image = cv2.inRange(scan, np.array(property.min_color), np.array(property.max_color))
        _, prop_image = cv2.threshold(prop_image, 127, 255, cv2.THRESH_BINARY)

        feature_vector.append(get_property_info(prop_image))

    #TODO: build feature vector
    return feature_vector

#TODO if main: given input folder, create JSON with all feature vectors

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--image_folder_name", help="path to read images")
    parser.add_argument("-l", "--label_file_name", help="path to left / right label json file")
    parser.add_argument("-o", "--output_file_name", help="path to write output json file")
    args = parser.parse_args()

    png_file_names = [f for f in listdir(args.image_folder_name) if isfile(join(args.image_folder_name, f)) and f.endswith('.png')]

    #load l/r labels
    labels = handle_json.json_file_to_obj(args.label_file_name)
    #Load template images
    for template in labels.templates:
        template.image = cv2.imread(template.file_name)

    property_dictionary = handle_json.json_file_to_obj('FeatureExtraction/scan_properties.json')

    feature_vectors = SimpleNamespace()
    feature_vectors.scan_features = []

    #temp: debugging
    max_files = 8
    for i in range(min(max_files, len(png_file_names))):

        scan_image = cv2.imread(join(args.image_folder_name, png_file_names[i]))

        #Remove correct left / right template from image
        label = next(label.label for label in labels.labels if label.file_name == png_file_names[i])
        template = next(template.image for template in labels.templates if template.label == label)
        scan_image = cv2.subtract(scan_image, template)
        #todo: infill missing areas

        cv2.imshow('Scan_image {}'.format(i),scan_image)

        feature_vectors.scan_features.append(get_feature_vector(scan_image, property_dictionary))

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    handle_json.obj_to_json_file(feature_vectors, args.output_file_name)
