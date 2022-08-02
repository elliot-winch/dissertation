import torch
from torchvision import transforms
import numpy as np

from types import SimpleNamespace
import argparse
import time

import AE_Architectures
import handle_json
import handle_dataloader
from progress_bar import log_progress_bar

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_folder", help="path to dataset")
    parser.add_argument("-e", "--encoder_path", help="path to encoder folder")
    args = parser.parse_args()

    #Load config
    config = handle_json.json_file_to_obj(args.encoder_path + '/info.json')

    #Load images
    tfs = []
    tfs.append(transforms.Resize(config.image_size))
    if config.greyscale:
        tfs.append(transforms.Grayscale(num_output_channels=1))
    tfs.append(transforms.ToTensor())

    image_transform = transforms.Compose(tfs)
    image_dataloader = handle_dataloader.create_dataloader(args.image_folder, image_transform, batch_size = 1)

    #Load model
    architecture = AE_Architectures.architectures[config.architecture_name]
    encoder = architecture.Encoder(config.image_size)
    model = torch.load(args.encoder_path + '/model_encoder.pth')
    encoder.load_state_dict(model)
    encoder.eval()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    encoder.to(device)

    file_contents = SimpleNamespace()
    file_contents.encoded_images = []
    image_dataloader_iter = iter(image_dataloader)
    with torch.no_grad():
        #With batch size = 1, each image is loaded one by one, so the file names match
        for i in range(len(image_dataloader)):
            log_progress_bar(i / len(image_dataloader))
            images, _ = next(image_dataloader_iter)
            images = images.to(device)
            encoded_data = encoder(images)

            entry = SimpleNamespace()
            entry.file_name = image_dataloader.dataset.samples[i][0]
            entry.feature_vector = encoded_data.cpu().detach().numpy().tolist()
            file_contents.encoded_images.append(entry)
        log_progress_bar(1)

    handle_json.obj_to_json_file(file_contents, "{}/encoded_images_{}.json".format(args.encoder_path, time.strftime("%m%d_%H%M%S")))
