import torch
from torch import nn

from image_size_calculations import image_size_after_convolution

encoded_space_dim = 50

class Encoder(nn.Module):

    def __init__(self, image_size):
        super().__init__()

        image_size = image_size_after_convolution(image_size, 3, 1, 2)
        image_size = image_size_after_convolution(image_size, 3, 1, 2)
        #image_size = image_size / 2 #max pooling
        image_size = image_size_after_convolution(image_size, 3, 1, 2)

        ### Convolutional layers
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(True)
        )

        # Flatten layer
        self.flatten = nn.Flatten(start_dim=1)

        # Linear layers
        self.lin = nn.Sequential(
            nn.Linear(image_size * image_size * 32, 128),
            nn.ReLU(True),
            nn.Linear(128, encoded_space_dim)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.flatten(x)
        x = self.lin(x)
        return x

class Decoder(nn.Module):

    def __init__(self, image_size):
        super().__init__()

        image_size = image_size_after_convolution(image_size, 3, 1, 2)
        image_size = image_size_after_convolution(image_size, 3, 1, 2)
        #image_size = image_size / 2 #max pooling
        image_size = image_size_after_convolution(image_size, 3, 1, 2)

        # 1 - Linear layers
        self.lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, image_size * image_size * 32),
        )

        # 2 - Flatten layer
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, image_size, image_size))

        ### 3 - Convolutional layers
        self.cnn = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(16),
            #nn.Upsample(scale_factor=1),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            #nn.Upsample(scale_factor=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 3, 3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, x):
        x = self.lin(x)
        x = self.unflatten(x)
        x = self.cnn(x)
        x = torch.sigmoid(x)
        return x
