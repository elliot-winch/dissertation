import cv2
import argparse
from os import listdir
from os.path import isfile, join
from progress_bar import log_progress_bar

def greyscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def remove_blue_channel(image):
    for i in range(len(image)):
        for j in range(len(image[0])):
            image[i, j] = [0, image[i, j, 1], image[i, j, 2]]
    return image

transformation = remove_blue_channel
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_folder", help="path to dataset")
    parser.add_argument("-o", "--output_folder", help="path to output folder")
    args = parser.parse_args()

    png_file_names = [f for f in listdir(args.image_folder) if isfile(join(args.image_folder, f)) and f.endswith('.png')]

    print("Creating...")
    transformed_images = []
    for i in range(len(png_file_names)):
        log_progress_bar(i / len(png_file_names))
        image = cv2.imread(join(args.image_folder, png_file_names[i]))
        cv2.imwrite(join(args.output_folder, png_file_names[i]), transformation(image))

    log_progress_bar(1)
