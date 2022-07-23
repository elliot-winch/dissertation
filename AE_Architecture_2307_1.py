import torch
from torch import nn

from image_size_calculations import image_sizes_after_convolutions

encoded_space_dim = 50

class Encoder(nn.Module):

    def __init__(self, image_size):
        super().__init__()

        image_sizes = image_sizes_after_convolutions(image_size, [3] * 3, [1] * 3, [2] * 3)

        ### Convolutional layers
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(True)
        )

        # Flatten layer
        self.flatten = nn.Flatten(start_dim=1)

        # Linear layers
        self.lin = nn.Sequential(
            nn.Linear(image_sizes[-1] * image_sizes[-1] * 32, 128),
            nn.ReLU(True),
            nn.Linear(128, encoded_space_dim)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.flatten(x)
        x = self.lin(x)
        return x

class Decoder(nn.Module):

    #Same input as the encoder for simplicity - channels need to be reversed
    #Reversing the input makes the cnn_image_sizes calculation messy as it would
    #require reversing reversed list.
    def __init__(self, image_size):
        super().__init__()

        image_sizes = image_sizes_after_convolutions(image_size, [3] * 3, [1] * 3, [2] * 3)

        # 1 - Linear layers
        self.lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, image_sizes[-1] * image_sizes[-1] * 32),
        )

        # 2 - Flatten layer
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, image_sizes[-1], image_sizes[-1]))

        ### 3 - Convolutional layers
        self.cnn = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 3, 3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, x):
        x = self.lin(x)
        x = self.unflatten(x)
        x = self.cnn(x)
        x = torch.sigmoid(x)
        return x
