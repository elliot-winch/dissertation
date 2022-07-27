import cv2
import argparse
from os import listdir
from os.path import isfile, join
from progress_bar import log_progress_bar

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_folder", help="path to dataset")
    parser.add_argument("-o", "--output_folder", help="path to output folder")
    args = parser.parse_args()

    png_file_names = [f for f in listdir(args.image_folder) if isfile(join(args.image_folder, f)) and f.endswith('.png')]

    greyscale_images = []
    for i in range(len(png_file_names)):
        image = cv2.imread(join(args.image_folder, png_file_names[i]))
        greyscale_images.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

    for i in range(len(png_file_names)):
        log_progress_bar(i / len(png_file_names))
        cv2.imwrite(join(args.output_folder, png_file_names[i]), greyscale_images[i])
    log_progress_bar(1)
