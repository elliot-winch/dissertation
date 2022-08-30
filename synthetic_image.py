import random
import torch
import cv2
import matplotlib.pyplot as plt

import argparse

import AE_Architectures
from lerp import lerp_vector
import handle_json
from convert_image import tensor_to_numpy

seed = 10

def load_decoder(encoder_path):
    config = handle_json.json_file_to_obj(encoder_path + '/info.json')
    architecture = AE_Architectures.architectures[config.architecture_name]
    decoder = architecture.Decoder(config.image_size)
    model = torch.load(encoder_path + '/model_decoder.pth')
    decoder.load_state_dict(model)
    decoder.eval()
    return decoder

#Testing
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--encoder_path", help="path to eoncder folder")
    parser.add_argument("-e", "--encoded_images", help="name of the encoded image json file")
    args = parser.parse_args()

    encoded_images = handle_json.json_file_to_obj(args.encoder_path + '/' + args.encoded_images).encoded_images

    decoder = load_decoder(args.encoder_path)

    #Choose two random images from all encoded images to test
    random.seed(seed)
    encoded_image_a = encoded_images[random.randint(0, len(encoded_images))]
    encoded_image_b = encoded_images[random.randint(0, len(encoded_images))]

    with torch.no_grad():
        #Generate synethic image
        lerp_quantity = 0.5
        synthetic_vector = lerp_vector(encoded_image_a.feature_vector, encoded_image_b.feature_vector, lerp_quantity)
        synthetic_vector = torch.Tensor(synthetic_vector)
        synthetic_vector = torch.unsqueeze(synthetic_vector, dim=0) #make a batch of 1
        synthetic_image = decoder(synthetic_vector)
        synthetic_image = tensor_to_numpy(synthetic_image[0])

        #Load actual images
        image_a = cv2.imread(encoded_image_a.file_name)
        image_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2RGB)
        image_b = cv2.imread(encoded_image_b.file_name)
        image_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2RGB)

        #Decode sample images
        #TODO: replace with convert_image.decode
        feature_vector_a = torch.Tensor(encoded_image_a.feature_vector)
        feature_vector_a = torch.unsqueeze(feature_vector_a, dim=0)
        decoded_image_a = decoder(feature_vector_a)
        decoded_image_a = tensor_to_numpy(decoded_image_a[0])

        feature_vector_b = torch.Tensor(encoded_image_b.feature_vector)
        feature_vector_b = torch.unsqueeze(feature_vector_b, dim=0)
        decoded_image_b = decoder(feature_vector_b)
        decoded_image_b = tensor_to_numpy(decoded_image_b[0])

        #Display all images
        f, axarr = plt.subplots(2,3)
        axarr[0,0].imshow(image_a)
        axarr[0,1].imshow(image_b)

        axarr[1,0].imshow(decoded_image_a)
        axarr[1,1].imshow(decoded_image_b)
        axarr[1,2].imshow(synthetic_image)
        plt.show()
