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
import torch.fft as fft

import argparse
import time
import os
from types import SimpleNamespace

import handle_dataloader
import handle_json
from progress_bar import log_progress_bar
import AE_Architectures

#Transform image tensor into matplotlib-displayable image
def tensor_to_plt(image):
    return np.transpose(image.cpu().detach().numpy(), (1,2,0))

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

    return val_loss.detach().cpu().numpy()

### Examine examples
def plot_ae_outputs(encoder, decoder, device, dataloader, n = 10):
    #Use the first batch as an example
    images, _ = next(iter(dataloader))
    with torch.no_grad():
        encoder.eval()
        decoder.eval()
        reconstructed_images = decoder(encoder(images.to(device)))

    plot_images(images, reconstructed_images, n)


def plot_images(images, reconstructed_images, n):

    plt.figure(figsize=(16,4.5))

    for i in range(min(len(images), n)):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(tensor_to_plt(images[i]), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        if i == n//2:
            ax.set_title('Original images')

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(tensor_to_plt(reconstructed_images[i]), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        if i == n//2:
            ax.set_title('Reconstructed images')
    #plt.show()

def plot_layer_images(images, layer_images, output_folder_name, layer_name):
    for i in range(layer_images.shape[1]): #1st dimension is number of channels
        channel_images = torch.index_select(layer_images, 1, torch.tensor([i]).to(device))
        plot_images(images, channel_images, n=10)
        plt.savefig(output_folder_name + "/{}_{}".format(layer_name, i))
        plt.clf()

"""
loss_func = torch.nn.MSELoss()
def freq_loss(output, target):
    output_freq = fft.fft2(output)
    target_freq = fft.fft2(target)
    return torch.mean(torch.abs(target_freq.abs() - output.abs()))
"""

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_folder", help="path to dataset")
    parser.add_argument("-a", "--architecture_name", help="name of file that defines encoder/decoder architecture")
    parser.add_argument("-o", "--output_folder", help="path to experiment output folder")
    parser.add_argument("-g", "--greyscale", help="are the input images in greyscale?", action="store_true")
    parser.add_argument("-s", "--save", help="save the model?", action="store_true")
    args = parser.parse_args()

    #todo: config file that allows for mutliple encoders
    batch_size = 32
    lr = 0.001
    weight_decay = 1e-05
    num_epochs = 15
    image_size = 128

    #Load architecture
    architecture = AE_Architectures.architectures[args.architecture_name]

    tfs = []
    tfs.append(transforms.Resize(image_size))
    if args.greyscale:
        tfs.append(transforms.Grayscale(num_output_channels=1))
    tfs.append(transforms.ToTensor())

    #Load data
    image_transform = transforms.Compose(tfs)
    train_loader = handle_dataloader.create_dataloader(args.image_folder + '/train', image_transform, batch_size = batch_size)
    val_loader = handle_dataloader.create_dataloader(args.image_folder + '/val', image_transform, batch_size = batch_size)
    test_loader = handle_dataloader.create_dataloader(args.image_folder + '/test', image_transform, batch_size = batch_size)

    #Prepare output
    #Errors will occur before training to avoid wasting time training when output is invalid
    output_folder_name = args.output_folder + '/' + args.architecture_name + time.strftime("%m%d_%H%M%S")
    os.mkdir(output_folder_name)

    output_info = SimpleNamespace()
    output_info.batch_size = batch_size
    output_info.lr = lr
    output_info.weight_decay = weight_decay
    output_info.num_epochs = num_epochs
    output_info.image_size = image_size
    output_info.greyscale = args.greyscale
    output_info.architecture_name = args.architecture_name

    ### Define the loss function
    loss_fn = torch.nn.MSELoss()

    ### Set the random seed for reproducible results
    torch.manual_seed(0)

    encoder = architecture.Encoder(image_size)
    decoder = architecture.Decoder(image_size)
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

    train_losses = []
    val_losses = []
    print("Running for {} epochs".format(num_epochs))
    for epoch in range(num_epochs):
        print("\nEpoch {}".format(epoch))
        print("Training...")
        train_loss = train_epoch(encoder, decoder, device, train_loader, loss_fn, optim)
        print("\nTesting...")
        val_loss = test_epoch(encoder, decoder, device, val_loader, loss_fn)
        print('\n Train loss: {} \t Val loss: {}'.format(train_loss, val_loss))
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    if args.save:
        torch.save(encoder.state_dict(), output_folder_name + '/model_encoder.pth')
        torch.save(decoder.state_dict(), output_folder_name + '/model_decoder.pth')

    output_info.train_losses = train_losses
    output_info.val_losses = val_losses

    #See example images from layers
    images, _ = next(iter(test_loader))
    images = images.to(device)
    with torch.no_grad():
        en_first_layer = encoder.first_layer(images)
        en_second_layer = encoder.second_layer(en_first_layer)
        en_third_layer = encoder.third_layer(en_second_layer)

        en_images = encoder(images)
        de_images = decoder.lin(en_images)
        de_images = decoder.unflatten(de_images)
        de_first_layer = decoder.first_layer(de_images)
        de_second_layer = decoder.second_layer(de_first_layer)
        de_third_layer = decoder.third_layer(de_second_layer)

    plot_max_layers = 10
    plot_layer_images(images[:plot_max_layers], en_first_layer[:plot_max_layers], output_folder_name, "En_1")
    #plot_layer_images(images[:plot_max_layers], en_second_layer[:plot_max_layers], output_folder_name, "En_2")
    #plot_layer_images(images[:plot_max_layers], en_third_layer[:plot_max_layers], output_folder_name, "En_3")
    #plot_layer_images(images[:plot_max_layers], de_first_layer[:plot_max_layers], output_folder_name, "De_1")
    #plot_layer_images(images[:plot_max_layers], de_second_layer[:plot_max_layers], output_folder_name, "De_2")
    plot_layer_images(images[:plot_max_layers], de_third_layer[:plot_max_layers], output_folder_name, "De_3")

    plot_ae_outputs(encoder, decoder, device, test_loader, n=10)
    plt.savefig(output_folder_name + '/AE_Test_Examples')

    handle_json.obj_to_json_file(output_info, output_folder_name + '/info.json')

    # Plot losses
    plt.figure(figsize=(10,8))
    plt.semilogy(train_losses, label='Train')
    plt.semilogy(val_losses, label='Valid')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    #plt.grid()
    plt.legend()
    #plt.title('loss')
    #plt.show()
