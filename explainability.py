import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors

import handle_json
import convert_image

from nearest_neighbours import get_nearest_neighbours
from lerp import lerp_vector
from decision_boundary import build_classifier
from synthetic_image import load_decoder
import confusion_matrix

def find_nearest(array, value, num_to_find = 1):
    order_by = lambda v : np.linalg.norm(np.array(v)- np.array(value))
    sorted_array = sorted(array, key=order_by)
    return sorted_array[0:num_to_find]

#NB only works with two classes
#start - encoded image vector and class
def counterfactual_image(start, target, classifier, iterations = 8):
    lower_bound = 0
    upper_bound = 1

    start_class = classifier.predict(np.array(start.feature_vector).reshape(1, -1))[0]
    target_class = classifier.predict(np.array(target).reshape(1, -1))[0]
    print(start_class, target_class)

    values = []
    for i in range(iterations):
        half_value = (upper_bound + lower_bound) / 2
        values.append(half_value)
        counterfactual = lerp_vector(start.feature_vector, target, half_value)

        #Predict requires 2D array - inputting as a single sample
        counterfactual_class = classifier.predict(np.array(counterfactual).reshape(1, -1))[0]
        if counterfactual_class == start.image_class:
            lower_bound = half_value
        else:
            upper_bound = half_value

    print(values, '\n\n')
    return counterfactual

class_to_examine = 0
skip = 1
num_to_produce = 5
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--encoder_path", help="path to encoder folder")
    parser.add_argument("-e", "--encoded_images", help="name of the encoded image json file")
    parser.add_argument("-c", "--classification_algorithm", help="name of classification algorithm to use")
    args = parser.parse_args()

    #Load file contents
    file_contents = handle_json.json_file_to_obj(args.encoder_path + '/' + args.encoded_images)
    encoded_images = file_contents.encoded_images

    #Build classifier. Split feature vectors and classes first
    classifier = build_classifier(encoded_images, args.classification_algorithm)
    predicted = classifier.predict([encoded_image.feature_vector for encoded_image in encoded_images])

    print(confusion_matrix.get_confusion_matrx([encoded_image.image_class for encoded_image in encoded_images], predicted, num_classes = file_contents.number_of_classes))

    selected_image = None
    #Fine the first correctly classified example for a class
    for i in range(len(encoded_images)):
        if predicted[i] == class_to_examine and encoded_images[i].image_class == class_to_examine:
            if skip <= 0:
                selected_image = encoded_images[i]
                break
            else:
                skip -= 1

    if selected_image == None:
        print("Failed to find any correctly classified examples of the class {}".format(class_to_examine))
        exit()

    #Calculate counterfactuals
    #Use predicted classes
    opposite_class_feature_vectors = [encoded_images[i].feature_vector for i in range(len(encoded_images)) if predicted[i] != class_to_examine]
    nearest_opposite = find_nearest(opposite_class_feature_vectors, selected_image.feature_vector, num_to_find=num_to_produce)

    counterfactuals = [counterfactual_image(selected_image, nearest_opposite[i], classifier) for i in range(len(nearest_opposite))]

    decoder = load_decoder(args.encoder_path)

    with torch.no_grad():

        #Display results
        original_image = convert_image.decode(selected_image.feature_vector, decoder)

        f, axarr = plt.subplots(2,1+num_to_produce)
        axarr[0,0].imshow(original_image)

        for i in range(len(counterfactuals)):
            target_image = convert_image.decode(nearest_opposite[i], decoder)
            axarr[0, i + 1].imshow(target_image)
            cf_image = convert_image.decode(counterfactuals[i], decoder)
            axarr[1, i + 1].imshow(cf_image)

        plt.show()
