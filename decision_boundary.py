from sklearn.cluster import KMeans
import numpy as np
from types import SimpleNamespace

import argparse

import handle_json
import confusion_matrix

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--encoded_images", help="name of the encoded image json file")
    parser.add_argument("-o", "--output_file", help="name of the output json file")
    args = parser.parse_args()

    file_contents = handle_json.json_file_to_obj(args.encoded_images)
    encoded_images = file_contents.encoded_images

    feature_vectors = [encoded_image.feature_vector for encoded_image in file_contents.encoded_images]
    classes = [encoded_image.image_class for encoded_image in file_contents.encoded_images]
    classes = [1 if class_ == 'Success' else 0 for class_ in classes]

    #todo: other decision boundaries
    kmeans = KMeans(n_clusters=2, random_state=0).fit(feature_vectors)

    print(classes)
    print(kmeans.labels_)

    output = SimpleNamespace()
    output.confusion_matrix = confusion_matrix.get_confusion_matrx(classes, kmeans.labels_, 2)

    handle_json.obj_to_json_file(output, args.output_file)
