import numpy as np
import matplotlib.pyplot as plt

import argparse

import handle_json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--encoded_images", help="name of the encoded image json file")
    args = parser.parse_args()

    file_contents = handle_json.json_file_to_obj(args.encoded_images)

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    colors = ['red', 'blue']

    for class_ in range(file_contents.number_of_classes):
        x_points = []
        y_points = []
        z_points = []
        for encoded_image in file_contents.encoded_images:
            if encoded_image.image_class == class_:
                x_points.append(encoded_image.feature_vector[0])
                y_points.append(encoded_image.feature_vector[1])
                z_points.append(encoded_image.feature_vector[2])

        ax.scatter3D(x_points, y_points, z_points, color=colors[class_], s=1)

    plt.show()
