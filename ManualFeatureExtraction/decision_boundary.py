from sklearn.cluster import KMeans

import argparse
import matplotlib.pyplot as plt

import handle_json
from confusion_matrix import get_confusion_matrx
from arrange_files import get_result_from_file_name
from progress_bar import log_progress_bar

class_names = ["Failure", "Success"]

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--features_file_name", help="path to features json")
    parser.add_argument("-o", "--output_file_name", help="path to write output json file")
    args = parser.parse_args()

    scans = handle_json.json_file_to_obj(args.features_file_name).scans

    max_num_features = 8
    feature_size = 1
    feature_vector_size = max_num_features * feature_size

    features = []
    truth = []

    for scan_index in range(len(scans)):
        log_progress_bar(float(scan_index) / len(scans))

        scan = scans[scan_index]
        result = get_result_from_file_name(scan.file_name)

        if result not in class_names:
            continue

        truth.append(class_names.index(result))

        feature_vector = [0 for i in range(feature_vector_size)]

        scan_features = next(property.features for property in scan.properties if property.name == "horseshoe_tears")

        for i in range(min(len(scan_features), max_num_features)):
            starting_index = i * feature_size
            feature_vector[starting_index] = int(scan_features[i].size)
            feature_vector[starting_index] = int(scan_features[i].distance)
            feature_vector[starting_index] = int(scan_features[i].rotation)
            #feature_vector[starting_index] = shape

        features.append(feature_vector)

    log_progress_bar(1)
    print("\nBegin decision boundary")

    clustering = KMeans(n_clusters=2)
    clustering.fit(features)

    conf_matrix = get_confusion_matrx(truth, clustering.labels_, 2)
    print(conf_matrix)
