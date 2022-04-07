import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import torch.optim as optim

import torchvision

import numpy as np

from alex_net import AlexNet

"""
Wrapped class for training & testing a Neural Network
"""
class NeuralNetwork():

    data_dir = 'retina_data'
    output_path = 'model.pth'
    torch_seed = 1
    image_size = 227
    batch_size = 32
    shuffle_after_epoch = True

    learning_rate = 1e-3
    momentum = 0.9
    epochs = 2
    debug_print_every = 200

    classes = [ 'Failure', 'Success' ]

    def train(self):

        torch.manual_seed(self.torch_seed)

        #Data loaders
        train_transform = transforms.Compose([transforms.Resize(self.image_size), transforms.ToTensor()])
        train_dataset = datasets.ImageFolder(self.data_dir, transform=train_transform)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle_after_epoch)

        """
        val_transform = transforms.Compose([transforms.Resize(self.image_size), transforms.ToTensor()])
        val_dataset = datasets.ImageFolder(self.data_dir, transform=val_transform)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=self.shuffle_after_epoch)
        """

        """
        images, labels = next(iter(train_dataloader))

        print(images.size())
        """

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Torch device in use: " + str(device))

        model = AlexNet()
        model = model.to(device=device) #to send the model for training on either cuda or cpu

        criterion = nn.CrossEntropyLoss() #Adjust this function to use different loss
        optimizer = optim.SGD(model.parameters(), lr=self.learning_rate, momentum=self.momentum)

        for epoch in range(self.epochs):  # loop over the dataset multiple times

            print("\n\nNew Epoch: " + str(epoch))

            running_loss = 0.0
            for i, data in enumerate(train_dataloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs, labels = inputs.cuda(), labels.cuda() # Send data to GPU

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % self.debug_print_every == 0:    # debug print every N mini-batches
                    print(f'[{epoch}, {i}] loss: {running_loss / self.debug_print_every:.5f}')
                    running_loss = 0.0

        torch.save(model.state_dict(), self.output_path)

    def test(self):


        test_transform = transforms.Compose([transforms.Resize(self.image_size), transforms.ToTensor()])
        test_dataset = datasets.ImageFolder(self.data_dir, transform=test_transform) #temp: using same data to test
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=self.shuffle_after_epoch)

        images, labels = iter(test_dataloader).next()

        # Debug: print images
        print('GroundTruth: ', ' '.join(f'{self.classes[labels[j]]:5s}' for j in range(self.batch_size)))

        net = AlexNet()
        net.load_state_dict(torch.load(self.output_path))

        outputs = net(images)
        _, predicted = torch.max(outputs, 1)

        print('Predicted: ', ' '.join(f'{self.classes[predicted[j]]:5s}' for j in range(self.batch_size)))

        self.imshow(torchvision.utils.make_grid(images))

    def imshow(self, img):
        #img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
