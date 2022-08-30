import torch
import numpy as np
from torchvision import transforms
import cv2

import matplotlib.pyplot as plt
import argparse

def tensor_to_numpy(tensor):
    return np.transpose(tensor.cpu().detach().numpy(), (1,2,0))

def numpy_to_tensor(array):
    tf = transforms.ToTensor()
    return tf(array) #np.transpose(array, (2,0,1)))

def decode(feature_vector, decoder):
    feature_vector = torch.Tensor(feature_vector)
    feature_vector = torch.unsqueeze(feature_vector, dim=0)
    decoded_image = decoder(feature_vector)
    return tensor_to_numpy(decoded_image[0])

#Testing
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--test_image", help="path to test image")
    args = parser.parse_args()

    image = cv2.imread(args.test_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #Shift range from 0 - 255 to 0 - 1
    image = image.astype(np.float32)

    plt.imshow(image)
    plt.show()

    conv_image = tensor_to_numpy(numpy_to_tensor(image))
    print(numpy_to_tensor(image)[:, 198, 237])

    plt.imshow(conv_image)
    plt.show()

    mse = (np.square(image.astype(int) - conv_image.astype(int))).mean(axis=None)
    print("MSE: {}".format(mse))
