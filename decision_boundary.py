from sklearn import svm
from sklearn.cluster import KMeans
import numpy as np
from types import SimpleNamespace

import argparse
import time

import handle_json
import confusion_matrix
import measure_performance

def build_classifier(encoded_images, classification_algorithm):
    feature_vectors = [encoded_image.feature_vector for encoded_image in encoded_images]
    classes = [encoded_image.image_class for encoded_image in encoded_images]

    if classification_algorithm == "kmeans":
        classifier = KMeans(n_clusters=2, random_state=0).fit(feature_vectors)
    elif classification_algorithm == "svm":
        classifier = svm.SVC().fit(feature_vectors, classes)
    return classifier

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--encoder_path", help="path to encoder folder")
    parser.add_argument("-e", "--encoded_images", help="name of the encoded image json file")
    parser.add_argument("-c", "--classification_algorithm", help="name of classification algorithm to use")
    args = parser.parse_args()

    file_contents = handle_json.json_file_to_obj(args.encoder_path + '\\' + args.encoded_images)
    encoded_images = file_contents.encoded_images

    feature_vectors = [encoded_image.feature_vector for encoded_image in file_contents.encoded_images]

    classifier = build_classifier(encoded_images, args.classification_algorithm)
    predicted = classifier.predict(feature_vectors)

    output = SimpleNamespace()
    output.algorithm = args.classification_algorithm
    output.confusion_matrix = confusion_matrix.get_confusion_matrx(classes, predicted, num_classes = file_contents.number_of_classes)
    output.accuracy = measure_performance.accuracy(output.confusion_matrix)
    output.false_positives = measure_performance.recall(output.confusion_matrix, row=0, _class=1)
    output.true_positives = measure_performance.recall(output.confusion_matrix, row=1, _class=1)

    output_file_name = args.encoder_path + '\\decision_boundary_{}'.format(time.strftime("%m%d_%H%M%S"))

    handle_json.obj_to_json_file(output, output_file_name)
