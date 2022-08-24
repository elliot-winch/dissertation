import torch
import random
from sklearn.neighbors import NearestNeighbors

from types import SimpleNamespace
import argparse
import time

import AE_Architectures
from lerp import lerp_vector
import handle_json
from convert_image import tensor_to_numpy

seed = 10
oversampling_factor = 2
nearest_neighbours_count = 10

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--encoder_path", help="path to encoder folder")
    parser.add_argument("-e", "--encoded_images", help="name of the encoded image json file")
    args = parser.parse_args()

    file_contents = handle_json.json_file_to_obj(args.encoder_path + '/' + args.encoded_images)

    random.seed(seed)

    feature_vectors_by_class = []
    nearest_neighbours_by_class = []

    for class_ in range(file_contents.number_of_classes):
        feature_vectors = []
        for encoded_image in file_contents.encoded_images:
            if encoded_image.image_class == class_:
                feature_vectors.append(encoded_image.feature_vector)

        feature_vectors_by_class.append(feature_vectors)
        nn = NearestNeighbors(n_neighbors=nearest_neighbours_count, algorithm='ball_tree').fit(feature_vectors)
        _, nearest_neighbours = nn.kneighbors(feature_vectors)
        nearest_neighbours_by_class.append(nearest_neighbours)

    with torch.no_grad():
        #For each class
        for class_ in range(len(nearest_neighbours_by_class)):
            feature_vectors = feature_vectors_by_class[class_]
            nearest_neighbours = nearest_neighbours_by_class[class_]
            #For each entry in the class
            for i in range(len(nearest_neighbours)):
                #Get the neighbours
                neighbours_indices = nearest_neighbours[i]
                #The first entry in the NN index list is the starting feature vector's index
                feature_vector = feature_vectors[neighbours_indices[0]]
                #For each neighbour
                for j in range(1, len(neighbours_indices)):
                    #Generate a synthetic image
                    lerp_quantity = random.random()
                    synthetic_vector = lerp_vector(feature_vector, feature_vectors[neighbours_indices[j]], lerp_quantity)
                    entry = SimpleNamespace()
                    entry.image_class = class_
                    entry.feature_vector = feature_vector
                    file_contents.encoded_images.append(entry)

    handle_json.obj_to_json_file(file_contents, "{}/synthetic_encoded_images_{}.json".format(args.encoder_path, time.strftime("%m%d_%H%M%S")))
