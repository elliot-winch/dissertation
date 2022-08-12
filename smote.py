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

#Testing
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--encoder_path", help="path to eoncder folder")
    parser.add_argument("-e", "--encoded_images", help="name of the encoded image json file")
    args = parser.parse_args()

    file_contents = handle_json.json_file_to_obj(args.encoder_path + '/' + args.encoded_images)

    #Choose two random images from all encoded images to test
    random.seed(seed)

    feature_vectors = [encoded_image.feature_vector for encoded_image in file_contents.encoded_images]
    nn = NearestNeighbors(n_neighbors=nearest_neighbours_count, algorithm='ball_tree').fit(feature_vectors)
    _, nearest_neighbours = nn.kneighbors(feature_vectors)

    with torch.no_grad():
        #For each vector
        for i in range(len(file_contents.encoded_images)):
            #Get the neighbours
            feature_vector = file_contents.encoded_images[i].feature_vector
            image_class = file_contents.encoded_images[i].image_class
            neighbours = nearest_neighbours[i]
            #For each neighbour
            for j in range(1, len(neighbours)):
                #Generate a synthetic image
                lerp_quantity = random.random()
                synthetic_vector = lerp_vector(feature_vector, feature_vectors[neighbours[j]], lerp_quantity)
                entry = SimpleNamespace()
                entry.image_class = image_class
                entry.feature_vector = feature_vector
                file_contents.encoded_images.append(entry)

    handle_json.obj_to_json_file(file_contents, "{}/synthetic_encoded_images_{}.json".format(args.encoder_path, time.strftime("%m%d_%H%M%S")))
