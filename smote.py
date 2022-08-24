import random
from sklearn.neighbors import NearestNeighbors

from types import SimpleNamespace
import argparse
import time

import AE_Architectures
from lerp import lerp_vector
import handle_json
from convert_image import tensor_to_numpy

def get_nearest_neighbours(feature_vectors, nearest_neighbours_count):
    nn = NearestNeighbors(n_neighbors=nearest_neighbours_count, algorithm='ball_tree').fit(feature_vectors)
    _, nearest_neighbours = nn.kneighbors(feature_vectors)
    return nearest_neighbours

def generate_synthetic_for_class(feature_vectors, nearest_neighbours, class_index):
    synthetic_vectors = []
    for i in range(len(nearest_neighbours)):
        #Get the neighbours
        neighbours_indices = nearest_neighbours[i]
        #The first entry in the NN index list is the starting feature vector's index
        feature_vector = feature_vectors[neighbours_indices[0]]
        #For each neighbour that isn't itself
        for j in range(1, len(neighbours_indices)):
            #Generate a synthetic image
            lerp_quantity = random.random()
            synthetic_vector = lerp_vector(feature_vector, feature_vectors[neighbours_indices[j]], lerp_quantity)
            entry = SimpleNamespace()
            entry.image_class = class_index
            entry.feature_vector = synthetic_vector
            entry.synthetic = True
            synthetic_vectors.append(entry)
    return synthetic_vectors

def generate_synthetic(encoded_images, oversampling_factor_by_class, seed = None):
    if seed is not None:
        random.seed(seed)

    synthetic_vectors = []
    for class_index in range(len(oversampling_factor_by_class)):
        if oversampling_factor_by_class[class_index] < 1:
            continue

        feature_vectors = [encoded_image.feature_vector for encoded_image in encoded_images if encoded_image.image_class == class_index]
        nearest_neighbours = get_nearest_neighbours(feature_vectors, oversampling_factor_by_class[class_index])
        synthetic_vectors = synthetic_vectors + generate_synthetic_for_class(feature_vectors, nearest_neighbours, class_index)
    return synthetic_vectors


seed = 1
oversampling_factor_by_class = [2, 0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--encoder_path", help="path to encoder folder")
    parser.add_argument("-e", "--encoded_images", help="name of the encoded image json file")
    #parser.add_argument("-s", "--seed", help="random int to seed random generator", type=int)

    args = parser.parse_args()

    file_contents = handle_json.json_file_to_obj(args.encoder_path + '/' + args.encoded_images)

    synthetic_vectors = generate_synthetic(file_contents.encoded_images, oversampling_factor_by_class)

    file_contents.encoded_images = file_contents.encoded_images + synthetic_vectors

    handle_json.obj_to_json_file(file_contents, "{}/synthetic_encoded_images_{}.json".format(args.encoder_path, time.strftime("%m%d_%H%M%S")))
