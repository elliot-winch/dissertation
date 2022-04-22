import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import torch.optim as optim

import torchvision

import numpy as np

import time

from ignite.metrics.confusion_matrix import ConfusionMatrix

from alex_net import AlexNet
from epoch_data import TrainingMetadata
import arrange_files

"""
Wrapped class for training & testing a Neural Network

#caution: each new model is ~0.2GB. Creating lots of models will quickly take up memory.
Ensure model name is the same as existing if you don't want to create a new file
"""
class NeuralNetwork(object):

    config = {}
    debug_print_every = 50

    def __init__(self, config):
        super(NeuralNetwork, self).__init__()
        self.config = config

    def train(self, needs_arrange=True):

        torch.manual_seed(self.config.seed)

        if needs_arrange:
            arrange_files.arrange_files(self.config)

        #Data loaders
        image_transform = transforms.Compose([transforms.Resize(self.config.image_size), transforms.ToTensor()])

        train_dataset = datasets.ImageFolder(self.config.sorted_data_dir + '/train', transform=image_transform)
        train_dataloader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)

        val_dataset = datasets.ImageFolder(self.config.sorted_data_dir + '/val', transform=image_transform)
        val_dataloader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=True)

        images, labels = next(iter(val_dataloader))

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = AlexNet()
        model = model.to(device=device) #to send the model for training on either cuda or cpu

        criterion = nn.CrossEntropyLoss() #Adjust this function to use different loss
        optimizer = optim.SGD(model.parameters(), lr=self.config.learning_rate, momentum=self.config.momentum)

        #Recording
        training_metadata = TrainingMetadata(self.config.epoch_data_path + time.strftime("%Y%m%d-%H%M%S"))

        for epoch_num in range(self.config.epochs):  # loop over the dataset N times
            print("\n\nNew Epoch: " + str(epoch_num))
            epoch_data = self.epoch(train_dataloader, model, criterion, optimizer)
            training_metadata.record(epoch_data[0], epoch_data[1])

        training_metadata.write()
        torch.save(model.state_dict(), self.config.model_path)

    def epoch(self, train_dataloader, model, criterion, optimizer):
        loss_sum = 0.0
        val_loss_sum = 0.0
        for i, data in enumerate(train_dataloader, 0):
            loss_sum += self.epoch_train(data, model, criterion, optimizer).item()
            val_loss_sum += self.epoch_val(data, model, criterion).item()

        loss = loss_sum / len(train_dataloader)
        validation_loss = val_loss_sum / len(train_dataloader)
        return [loss, validation_loss]

    def epoch_val(self, data, model, criterion):

        model.eval()

        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda() # Send data to GPU

        outputs = model(inputs)
        return criterion(outputs, labels)

    def epoch_train(self, data, model, criterion, optimizer):

        model.train()

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
        return loss

    def test(self):
        test_transform = transforms.Compose([transforms.Resize(self.config.image_size), transforms.ToTensor()])
        test_dataset = datasets.ImageFolder(self.config.sorted_data_dir + '/test', transform=test_transform)
        test_dataloader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=True)

        return self.confusion_matrix(test_dataloader)

    def confusion_matrix(self, dataloader):

        model = AlexNet()
        model.load_state_dict(torch.load(self.config.model_path))
        y_pred = []
        y_true = []

        # iterate over test data
        for inputs, labels in dataloader:
            output = model(inputs) # Feed Network

            output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
            y_pred.extend(output) # Save Prediction

            labels = labels.data.cpu().numpy()
            y_true.extend(labels) # Save Truth

        num_classes = len(self.config.classes)
        matrix = [ [0]*num_classes for i in range(num_classes)]

        for i in range(len(y_true)):
            matrix[y_true[i]][y_pred[i]] += 1

        return matrix

    def mean_average_precision(self, matrix):
        #matrix should be square
        total = 0
        total_correct = 0
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                total += matrix[i][j]

                if i is j:
                    total_correct += matrix[i][j]

        return total_correct / float(total)

    def imshow(self, img):
        #img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
