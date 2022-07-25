import cv2
import numpy as np
from math import sqrt, pi

import argparse
from os import listdir, makedirs
from os.path import isfile, join, dirname, exists

from types import SimpleNamespace
import handle_json
from progress_bar import log_progress_bar

#todo: calculate
image_center = [199.5, 199.5]
#It's likely these are extra pixels and not actual features
min_size = 100

#property_image: greyscale image of single property (e.g. horseshoe tears)
def get_features_for_property(property_image, property):
    num_labels, component_labeled_image, stats, centroids = cv2.connectedComponentsWithStats(property_image)

    properties_info = []

    #Label 0 is the background - pixels not within a feature
    for i in range(1, num_labels):
        component_image = cv2.inRange(component_labeled_image, i, i)

        size = stats[i, cv2.CC_STAT_AREA]

        if size < min_size:
            continue

        property_info = SimpleNamespace()
        property_info.size = size
        # can also access width and height from stats if needed

        relative_centroid = [centroids[i][0] - image_center[0], centroids[i][1] - image_center[1]]

        property_info.distance = sqrt (relative_centroid[0] * relative_centroid[0] + relative_centroid[1] * relative_centroid[1])
        #convert to clockwise by negating x coordinate
        angle = np.arctan2(relative_centroid[1], -relative_centroid[0])
        #with midnight being 0 radians
        property_info.rotation = (angle + (3 * pi / 2)) % (pi * 2)

        moments = cv2.moments(component_image)

        hu_moments = cv2.HuMoments(moments).tolist()
        #flatten array
        hu_moments = [x for xs in hu_moments for x in xs]
        property_info.hu_moments = hu_moments

        properties_info.append(property_info)

    #Only consider largest N features
    order_by = lambda prop : prop.size
    properties_info = sorted(properties_info, key=order_by, reverse=True)
    properties_info = properties_info[0:property.max_count]

    return properties_info


def get_feature_vector(scan, properties):
    properties_object = []

    for property in properties:
        property_object = SimpleNamespace()
        property_object.name = property.name

        prop_image = cv2.inRange(scan, np.array(property.color), np.array(property.color))
        property_object.features = get_features_for_property(prop_image, property)
        properties_object.append(property_object)

    return properties_object


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

    #load and order properties
    properties = handle_json.json_file_to_obj('FeatureExtraction/scan_properties.json').properties
    order_by = lambda prop : prop.order
    properties = sorted(properties, key=order_by, reverse=True)

    #init output
    feature_vectors = SimpleNamespace()
    feature_vectors.scans = []

    log_progress_bar(0)
    for i in range(len(png_file_names)):

        image_feature = SimpleNamespace()
        image_feature.file_name = png_file_names[i]

        scan_image = cv2.imread(join(args.image_folder_name, png_file_names[i]))

        #Remove correct left / right template from image
        #Find label for image
        label = next(label.label for label in labels.labels if label.file_name == png_file_names[i])
        #Find template image for label
        template = next(template.image for template in labels.templates if template.label == label)
        #Remove the template from the image
        scan_image = cv2.bitwise_and(scan_image, scan_image, mask=cv2.bitwise_not(template[:,:,2]))
        #in fill the removed template
        scan_image = cv2.inpaint(scan_image, template[:,:,2], 3, cv2.INPAINT_NS)

        image_feature.properties = get_feature_vector(scan_image, properties)
        feature_vectors.scans.append(image_feature)
        log_progress_bar(float(i) / len(png_file_names))


    log_progress_bar(1)
    handle_json.obj_to_json_file(feature_vectors, args.output_file_name)
