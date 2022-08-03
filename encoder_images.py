import torch
from torchvision import transforms
import numpy as np
import cv2

from types import SimpleNamespace
import argparse
import time

import AE_Architectures
import handle_json
import handle_image_loading
from progress_bar import log_progress_bar
import convert_image

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_folder", help="path to dataset")
    parser.add_argument("-e", "--encoder_path", help="path to encoder folder")
    args = parser.parse_args()

    #Load config
    config = handle_json.json_file_to_obj(args.encoder_path + '/info.json')

    #Load images
    file_paths = handle_image_loading.load_images(args.image_folder, return_full_path=True)

    #Create Transform
    tfs = []
    tfs.append(transforms.Resize(config.image_size))
    if config.greyscale:
        tfs.append(transforms.Grayscale(num_output_channels=1))
    #tfs.append(transforms.ToTensor())

    image_transform = transforms.Compose(tfs)

    #Load model
    architecture = AE_Architectures.architectures[config.architecture_name]
    encoder = architecture.Encoder(config.image_size)
    model = torch.load(args.encoder_path + '/model_encoder.pth')
    encoder.load_state_dict(model)
    encoder.eval()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    encoder.to(device)

    #Encode images
    file_contents = SimpleNamespace()
    file_contents.encoded_images = []
    with torch.no_grad():
        for i in range(len(file_paths)):
            log_progress_bar(i / len(file_paths))

            image = cv2.imread(file_paths[i])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image_tensor = convert_image.numpy_to_tensor(image)
            image_tensor = image_transform(image_tensor)
            image_tensor = image_tensor.to(device)
            #Make the image a batch tensor of size 1
            image_tensor = torch.unsqueeze(image_tensor, dim=0)
            encoded_data = encoder(image_tensor)

            entry = SimpleNamespace()
            entry.file_name = file_paths[i]
            entry.feature_vector = encoded_data.cpu().detach().numpy().tolist()
            file_contents.encoded_images.append(entry)
        log_progress_bar(1)

    handle_json.obj_to_json_file(file_contents, "{}/encoded_images_{}.json".format(args.encoder_path, time.strftime("%m%d_%H%M%S")))
