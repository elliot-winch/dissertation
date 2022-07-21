import matplotlib.pyplot as plt # plotting library
import numpy as np # this module is useful to work with numerical arrays
import pandas as pd
import random
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader,random_split
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

import argparse

import handle_dataloader
from progress_bar import log_progress_bar

class Encoder(nn.Module):

    def __init__(self, image_size, conv_layer_channels, linear_layer_sizes):
        super().__init__()

        #Maybe have parameterisable padding / stride / kernel_size
        kernel_size = 3
        padding = 1
        stride = 2

        image_sizes = cnn_image_sizes(image_size, len(conv_layer_channels), kernel_size, padding, stride)

        ### Convolutional layers
        cnn_modules = []
        for i in range(1, len(conv_layer_channels)):
            cnn_modules.append(
                nn.Sequential(
                    nn.Conv2d(conv_layer_channels[i - 1],
                              out_channels=conv_layer_channels[i],
                              kernel_size = kernel_size,
                              stride = stride,
                              padding = padding),
                    nn.BatchNorm2d(conv_layer_channels[i]),
                    nn.LeakyReLU())
            )

        self.cnn = nn.Sequential(*cnn_modules)

        # Flatten layer
        self.flatten = nn.Flatten(start_dim=1)

        # Linear layers
        linear_modules = []
        first_linear_layer_size = image_sizes[-1] * image_sizes[-1] * conv_layer_channels[-1]
        linear_layer_sizes.insert(0, first_linear_layer_size)

        for i in range(1, len(linear_layer_sizes)):
            linear_modules.append(nn.Sequential(
                nn.Linear(linear_layer_sizes[i - 1], linear_layer_sizes[i]),
                nn.ReLU(True)
            ))

        self.lin = nn.Sequential(*linear_modules)

    def forward(self, x):
        x = self.cnn(x)
        x = self.flatten(x)
        x = self.lin(x)
        return x

class Decoder(nn.Module):

    #Same input as the encoder for simplicity - channels need to be reversed
    #Reversing the input makes the cnn_image_sizes calculation messy as it would
    #require reversing reversed list.
    def __init__(self, image_size, conv_layer_channels, linear_layer_sizes):
        super().__init__()

        #Maybe have parameterisable padding / stride / kernel_size
        kernel_size = 3
        padding = 1
        stride = 2

        image_sizes = cnn_image_sizes(image_size, len(conv_layer_channels), kernel_size, padding, stride)

        # 1 - Linear layers
        linear_modules = []
        first_linear_layer_size = image_sizes[-1] * image_sizes[-1] * conv_layer_channels[-1]
        linear_layer_sizes.insert(0, first_linear_layer_size)

        linear_layer_sizes.reverse()
        for i in range(1, len(linear_layer_sizes)):
            linear_modules.append(nn.Sequential(
                nn.Linear(linear_layer_sizes[i - 1], linear_layer_sizes[i]),
                nn.ReLU(True)
            ))

        self.lin = nn.Sequential(*linear_modules)

        # 2 - Flatten layer
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(conv_layer_channels[-1], image_sizes[-1], image_sizes[-1]))

        ### 3 - Convolutional layers
        cnn_modules = []
        conv_layer_channels.reverse()
        for i in range(1, len(conv_layer_channels)):
            cnn_modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(conv_layer_channels[i - 1],
                              out_channels=conv_layer_channels[i],
                              kernel_size = kernel_size,
                              stride = stride,
                              padding = padding,
                              output_padding = 1),
                    nn.BatchNorm2d(conv_layer_channels[i]),
                    nn.LeakyReLU())
            )

        self.cnn = nn.Sequential(*cnn_modules)

    def forward(self, x):
        x = self.lin(x)
        x = self.unflatten(x)
        x = self.cnn(x)
        x = torch.sigmoid(x)
        return x

#TODO: if needed, adapt for different kernel_size, padding, stride at different layers
def cnn_image_sizes(image_size, num_layers, kernel_size, padding, stride):
    image_sizes = []
    for i in range(num_layers - 1):
        image_size = int((image_size + padding - int(kernel_size / 2)) / stride)
        image_sizes.append(image_size)

    return image_sizes

#Transform image tensor into matplotlib-displayable image
def numpy_to_plt(image):
    return np.transpose(image.cpu().detach().numpy(), (1, 2, 0))

### Training function
def train_epoch(encoder, decoder, device, dataloader, loss_fn, optimizer):
    # Set train mode for both the encoder and the decoder
    encoder.train()
    decoder.train()
    train_loss = []

    train_num_batches = float(len(dataloader))
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for i, data in enumerate(dataloader, 0):

        log_progress_bar(i / train_num_batches)

        image_batch, _ = data  # with "_" we just ignore the labels (the second element of the dataloader tuple)
        # Move tensor to the proper device
        image_batch = image_batch.to(device)
        # Encode data
        encoded_data = encoder(image_batch)
        # Decode data
        decoded_data = decoder(encoded_data)
        # Evaluate loss
        loss = loss_fn(decoded_data, image_batch)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        #print('\t partial train loss (single batch): %f' % (loss.data))
        train_loss.append(loss.detach().cpu().numpy())

    log_progress_bar(1)
    return np.mean(train_loss)

### Testing function
def test_epoch(encoder, decoder, device, dataloader, loss_fn):
    # Set evaluation mode for encoder and decoder
    encoder.eval()
    decoder.eval()
    with torch.no_grad(): # No need to track the gradients
        # Define the lists to store the outputs for each batch
        conc_out = []
        conc_label = []
        for image_batch, _ in dataloader:
            # Move tensor to the proper device
            image_batch = image_batch.to(device)
            # Encode data
            encoded_data = encoder(image_batch)
            # Decode data
            decoded_data = decoder(encoded_data)
            # Append the network output and the original image to the lists
            conc_out.append(decoded_data.cpu())
            conc_label.append(image_batch.cpu())
        # Create a single tensor with all the values in the lists
        conc_out = torch.cat(conc_out)
        conc_label = torch.cat(conc_label)
        # Evaluate global loss
        val_loss = loss_fn(conc_out, conc_label)
    return val_loss.data

### Examine examples
def plot_ae_outputs(encoder, decoder, device, dataloader, n = 10):

    plt.figure(figsize=(16,4.5))

    #Use the first batch as an example
    images, _ = next(iter(dataloader))
    with torch.no_grad():
        encoder.eval()
        decoder.eval()
        reconstructed_images = decoder(encoder(images.to(device)))

    for i in range(min(len(images), n)):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(numpy_to_plt(images[i]), cmap='gist_gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        if i == n//2:
            ax.set_title('Original images')

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(numpy_to_plt(reconstructed_images[i]), cmap='gist_gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        if i == n//2:
            ax.set_title('Reconstructed images')
    plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_folder", help="path to dataset")

    #parser.add_argument("-o", "--output_file_name", help="path to write output json file")
    args = parser.parse_args()

    #todo: config file that allows for mutliple encoders
    batch_size = 32
    lr = 0.001
    weight_decay = 1e-05
    num_epochs = 10
    image_size = 64

    image_transform = handle_dataloader.default_image_transform(image_size=image_size)
    train_loader = handle_dataloader.create_dataloader(args.image_folder + '/train', image_transform, batch_size = batch_size)
    val_loader = handle_dataloader.create_dataloader(args.image_folder + '/val', image_transform, batch_size = batch_size)
    test_loader = handle_dataloader.create_dataloader(args.image_folder + '/test', image_transform, batch_size = batch_size)

    ### Define the loss function
    loss_fn = torch.nn.MSELoss()

    ### Set the random seed for reproducible results
    torch.manual_seed(0)

    encoder = Encoder(image_size, [3, 8, 16], [1000, 100])
    decoder = Decoder(image_size, [3, 8, 16], [1000, 100])
    params_to_optimize = [
        {'params': encoder.parameters()},
        {'params': decoder.parameters()}
    ]

    optim = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=weight_decay)

    # Check if the GPU is available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Selected device: {device}')

    # Move both the encoder and the decoder to the selected device
    encoder.to(device)
    decoder.to(device)

    diz_loss = {'train_loss':[],'val_loss':[]}
    for epoch in range(num_epochs):
        print("\nEpoch {}".format(epoch))
        print("Training...")
        train_loss = train_epoch(encoder, decoder, device,train_loader, loss_fn, optim)
        print("\nTesting...")
        val_loss = test_epoch(encoder, decoder, device, val_loader, loss_fn)
        #print('\n EPOCH {}/{} \t train loss {} \t val loss {}'.format(epoch + 1, num_epochs, train_loss, val_loss))
        diz_loss['train_loss'].append(train_loss)
        diz_loss['val_loss'].append(val_loss)


    plot_ae_outputs(encoder, decoder, device, test_loader, n=10)

    test_epoch(encoder, decoder, device, test_loader, loss_fn).item()

    # Plot losses
    plt.figure(figsize=(10,8))
    plt.semilogy(diz_loss['train_loss'], label='Train')
    plt.semilogy(diz_loss['val_loss'], label='Valid')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    #plt.grid()
    plt.legend()
    #plt.title('loss')
    plt.show()
